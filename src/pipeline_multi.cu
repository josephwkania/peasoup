#include <data_types/timeseries.hpp>
#include <data_types/fourierseries.hpp>
#include <data_types/header.hpp>
#include <data_types/candidates.hpp>
#include <data_types/filterbank.hpp>
#include <data_types/series_file.hpp>
#include <transforms/dedisperser.hpp>
#include <transforms/resampler.hpp>
#include <transforms/folder.hpp>
#include <transforms/ffter.hpp>
#include <transforms/dereddener.hpp>
#include <transforms/spectrumformer.hpp>
#include <transforms/birdiezapper.hpp>
#include <transforms/peakfinder.hpp>
#include <transforms/distiller.hpp>
#include <transforms/harmonicfolder.hpp>
#include <transforms/scorer.hpp>
#include <transforms/template_bank_reader.hpp>
#include <utils/exceptions.hpp>
#include <utils/utils.hpp>
#include <utils/stats.hpp>
#include <utils/stopwatch.hpp>
#include <utils/cmdline.hpp>
#include <utils/output_stats.hpp>
#include <utils/progress_monitor.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cufft.h"
#include "pthread.h"
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <climits>
#include <algorithm>
#include <utility>
#include <optional>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>


typedef float DedispOutputType;

class DMDispenser {
private:
  DispersionTrials<DedispOutputType>& trials;
  pthread_mutex_t mutex;
  int dm_idx;
  int count;
  ProgressMonitor* prog;

public:
  DMDispenser(DispersionTrials<DedispOutputType>& trials, ProgressMonitor* pm = nullptr)
    :trials(trials),dm_idx(0), prog(pm){
    count = trials.get_count();
    pthread_mutex_init(&mutex, NULL);
  }

  int get_dm_trial_idx(void){
    pthread_mutex_lock(&mutex);
    int retval;
    if (dm_idx >= trials.get_count()){
      retval =  -1;
    } else {
      retval = dm_idx;
      dm_idx++;
      if (prog) prog->tick_dm();
    }
    pthread_mutex_unlock(&mutex);
    return retval;
  }

  virtual ~DMDispenser(){
    pthread_mutex_destroy(&mutex);
  }
};

// The distillers' fractional frequency tolerance for a window of WINDOW_SIZE
// samples, from the whole series' (SIZE samples): the same in Fourier bins
static float window_freq_tol(float freq_tol, unsigned int size, unsigned int window_size){
  return freq_tol * (float)size / (float)window_size;
}

// One searched span of each DM trial: the whole series (window -1) or one
// window of it, with its own FFTs, whitening, peak finding and candidates.
struct Scale {
  unsigned int size;   // FFT length
  std::size_t start;   // first sample
  int window;          // index among the windows, -1 for the whole series
  AccelerationPlan& acc_plan;
  CuFFTerR2C r2cfft;
  CuFFTerC2R c2rfft;
  DeviceFourierSeries<cufftComplex> d_fseries;
  ReusableDeviceTimeSeries<float, DedispOutputType> d_tim;
  DeviceTimeSeries<float> d_tim_resampled;
  DevicePowerSpectrum<float> d_pspec;
  Dereddener rednoise;
  SpectrumFormer former;
  PeakFinder cand_finder;
  HarmonicSums<float> sums;
  HarmonicFolder harm_folder;
  HarmonicDistiller harm_finder;
  AccelerationDistiller acc_still;
  float mean, std, rms;

  // FREQ_TOL: the fractional tolerance of the distillers, args.freq_tol for the
  // whole series and that times its length over a window's for a window, so
  // that it spans the same number of Fourier bins
  Scale(unsigned int size, std::size_t start, int window, AccelerationPlan& acc_plan,
        CmdLineOptions& args, float tsamp, float freq_tol)
    :size(size), start(start), window(window), acc_plan(acc_plan),
     r2cfft(size), c2rfft(size),
     d_fseries(size/2+1, 1.0/(size*tsamp)),
     d_tim(size), d_tim_resampled(size), d_pspec(d_fseries),
     rednoise(size/2+1),
     cand_finder(args.min_snr,args.min_freq,args.max_freq,size),
     sums(d_pspec,args.nharmonics),
     harm_folder(sums, args.single_precision_harmonic_sums),
     harm_finder(freq_tol,args.max_harm,false),
     acc_still(size*tsamp,freq_tol,true),
     mean(0), std(0), rms(0) {}
};

class Worker {
private:
  DispersionTrials<DedispOutputType>& trials;
  DMDispenser& manager;
  CmdLineOptions& args;
  AccelerationPlan& acc_plan;
  ProgressMonitor* prog; // nullptr → no progress bar
  unsigned int size;
  int device;
  std::map<std::string,Stopwatch> timers;
  std::optional<Keplerian_TemplateBank_Reader>& keplerian_tb;
  bool elliptical_orbit_search;
  const std::vector<std::size_t>& window_starts;
  unsigned int window_size;
  AccelerationPlan* window_acc_plan;
  std::size_t dm_offset; // index of this gulp's first DM trial in the full DM list
  std::vector<std::unique_ptr<Scale>> scales; // the whole series, then each window
  std::unique_ptr<Zapper> bzap;

  void prepare(void){
    if (!scales.empty())
      return;
    float tsamp = trials.get_tsamp();
    scales.emplace_back(new Scale(size, 0, -1, acc_plan, args, tsamp, args.freq_tol));
    for (std::size_t w = 0; w < window_starts.size(); ++w)
      scales.emplace_back(new Scale(window_size, window_starts[w], (int)w, *window_acc_plan, args, tsamp,
                                    window_freq_tol(args.freq_tol, size, window_size)));
    if (!args.zapfilename.empty()) {
      if (args.verbose)
        std::cout << "Using zapfile: " << args.zapfilename << "\n";
      bzap.reset(new Zapper(args.zapfilename));
    }
  }

  // The samples of TIM a scale searches
  void scale_view(Scale& s, DedispersedTimeSeries<DedispOutputType>& tim,
                  DedispersedTimeSeries<DedispOutputType>& view){
    std::size_t n = tim.get_nsamps() > s.start ? tim.get_nsamps() - s.start : 0;
    view.set_data(tim.get_data() + s.start);
    view.set_nsamps(std::min<std::size_t>(n, s.size));
    view.set_tsamp(tim.get_tsamp());
    view.set_dm(tim.get_dm());
  }

  void preprocess_time_series(DedispersedTimeSeries<DedispOutputType>& tim,
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim, unsigned int size) {
        if (args.verbose) {
            std::cout << "Copying time series to device (DM=" << tim.get_dm() << ")\n";
            std::cout << "Transferring " << tim.get_nsamps() << " samples\n";
        }
        d_tim.copy_from_host(tim);
        if (args.verbose) std::cout << "Copy from host complete\n";
        if (args.verbose) std::cout << "Removing baseline\n";
        d_tim.remove_baseline(std::min(tim.get_nsamps(), d_tim.get_nsamps()));
        if (args.verbose) std::cout << "Baseline removed\n";
        if (size > tim.get_nsamps()) {
            if (args.verbose) std::cout << "Padding with zeros\n";
            d_tim.fill(tim.get_nsamps(), d_tim.get_nsamps(), 0);
        }
        if (args.verbose) std::cout << "Preprocessing done\n";
    }
  // 2) FFT -> rednoise + zap + stats

  void remove_rednoise_and_zap(ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    CuFFTerR2C& r2cfft,
    CuFFTerC2R& c2rfft,
    DeviceFourierSeries<cufftComplex>& d_fseries,
    DevicePowerSpectrum<float>& d_pspec,
    Dereddener& rednoise,
    Zapper* bzap,
    SpectrumFormer& former,
    float& mean, float& rms, float& std, unsigned int size) {

    if (args.verbose) std::cout << "Executing forward FFT\n";
    r2cfft.execute(d_tim.get_data(), d_fseries.get_data());
    if (args.verbose) std::cout << "Forming power spectrum\n";
    former.form(d_fseries, d_pspec);
    if (args.verbose) std::cout << "Calculating running median\n";
    rednoise.calculate_median(d_pspec);
    if (args.verbose) std::cout << "Dereddening Fourier series\n";
    rednoise.deredden(d_fseries);
    if (bzap) {
        if (args.verbose) std::cout << "Zapping birdies\n";
        bzap->zap(d_fseries);
    }
    if (args.verbose) std::cout << "Forming interpolated spectrum\n";
    former.form_interpolated(d_fseries, d_pspec);
    if (args.verbose) std::cout << "Computing stats\n";
    //stats::stats<float>(d_pspec.get_data(), d_pspec.get_nbins(), &mean, nullptr, &std);
    //Check later if d_pspec.get_nbins() = size/2+1 (should be true)
    stats::stats<float>(d_pspec.get_data(), size/2+1, &mean, &rms, &std);
    if (args.verbose) std::cout << "Inverse FFT\n";
    c2rfft.execute(d_fseries.get_data(), d_tim.get_data());
    if (args.verbose) std::cout << "Rednoise removal and zapping complete\n";

    }

  
void run_search_and_find_candidates(DeviceTimeSeries<float>& d_tim_resampled,
    CuFFTerR2C& r2cfft,
    CuFFTerC2R& c2rfft,
    DeviceFourierSeries<cufftComplex>& d_fseries,
    DevicePowerSpectrum<float>& d_pspec,
    SpectrumFormer& former,
    HarmonicSums<float>& sums,
    HarmonicFolder& harm_folder,
    PeakFinder& cand_finder,
    HarmonicDistiller& harm_finder,
    SpectrumCandidates& trial_cands,
    CandidateCollection& output_cands,
    float mean, float std, unsigned int size) {

    if (args.verbose) std::cout << "Executing forward FFT\n";
    r2cfft.execute(d_tim_resampled.get_data(), d_fseries.get_data());
    if (args.verbose) std::cout << "Forming interpolated power spectrum\n";
    former.form_interpolated(d_fseries, d_pspec);
    if (args.verbose) std::cout << "Normalising power spectrum\n";
    stats::normalise(d_pspec.get_data(), mean * size, std * size, size / 2 + 1);
    if (args.verbose) std::cout << "Harmonic summing\n";
    harm_folder.fold(d_pspec);
    if (args.verbose) std::cout << "Finding peaks\n";
    cand_finder.find_candidates(d_pspec, trial_cands);
    cand_finder.find_candidates(sums, trial_cands);
    if (args.verbose) std::cout << "Distilling harmonics\n";
    output_cands.append(harm_finder.distill(trial_cands.cands));
}

void run_acceleration_time_domain_resampler(
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled,
    float acc, unsigned int size) {
    if (args.verbose) std::cout << "Resampling to " << acc << " m/s/s\n";
    TimeDomainResampler resampler;
    resampler.resampleII(d_tim, d_tim_resampled, size, acc);
    if (args.verbose) std::cout << "Resampling complete\n"; 
}


void run_acceleration_search(int idx,
    AccelerationPlan& acc_plan,
    CandidateCollection& accel_search_cands,
    std::vector<float>& acc_list,
    DedispersedTimeSeries<DedispOutputType>& tim,
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled,
    CuFFTerR2C& r2cfft,
    CuFFTerC2R& c2rfft,
    DeviceFourierSeries<cufftComplex>& d_fseries,
    DevicePowerSpectrum<float>& d_pspec,
    SpectrumFormer& former,
    HarmonicSums<float>& sums,
    HarmonicFolder& harm_folder,
    PeakFinder& cand_finder,
    HarmonicDistiller& harm_finder,
    float mean, float std, unsigned int size, bool tick = true
)
{
    if (args.verbose) std::cout << "Generating acceleration list\n";
    acc_plan.generate_accel_list(tim.get_dm(), args.cdm, acc_list);
    if (args.verbose) std::cout << "Searching " << acc_list.size() << " acceleration trials for DM " << tim.get_dm() << "\n";
    
    PUSH_NVTX_RANGE("Acceleration-Loop",1)

    for (int jj = 0; jj < acc_list.size(); jj++) {
        run_acceleration_time_domain_resampler(d_tim, d_tim_resampled, acc_list[jj], size);
        SearchParams accel_search;
        accel_search.acc = acc_list[jj];
        SpectrumCandidates trial_cands(tim.get_dm(), idx, accel_search);
        run_search_and_find_candidates(
            d_tim_resampled, r2cfft, c2rfft, d_fseries, d_pspec, former,
            sums, harm_folder, cand_finder, harm_finder, trial_cands,
            accel_search_cands, mean, std, size);
        // Update progress bar
        if (prog && tick) prog->tick_bin(); 
    }
    POP_NVTX_RANGE
}

void run_circular_orbit_search_resampler(
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled, unsigned int size,
    double n, double a1, double phi,  double tsamp, double inverse_tsamp) {
    if (args.verbose) std::cout << "Resampling to circular orbit with n=" << n << ", a1=" << a1 << ", phi=" << phi << "\n";
    TimeDomainResampler resampler;
    resampler.circular_orbit_resampler(d_tim, d_tim_resampled, size, n, a1, phi, tsamp, inverse_tsamp);
    if (args.verbose) std::cout << "Resampling complete\n"; 
}

void run_elliptical_orbit_search_resampler(
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled, unsigned int size,
    double n, double a1, double phi, double omega, double ecc,
    double tsamp, double inverse_tsamp) {
    if (args.verbose) std::cout << "Resampling to elliptical orbit with n=" << n << ", a1=" << a1 << ", phi=" << phi << ", omega=" << omega << ", ecc=" << ecc << "\n";
    TimeDomainResampler resampler;

    if (ecc >= 0.8) {
        if (args.verbose) std::cout << "Using BT model resampler for high eccentricity orbit\n";
        resampler.bt_model_resampler(d_tim, d_tim_resampled, n, a1, phi, omega, ecc, tsamp, inverse_tsamp, size);
    }
    else {
        if (args.verbose) std::cout << "Using ELL8 resampler for low eccentricity orbit\n";
        resampler.ell8_resampler(d_tim, d_tim_resampled, n, a1, phi, omega, ecc, tsamp, inverse_tsamp, size);
       
    }
}

void exact_resampler_elliptical(
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled, unsigned int size,
    double n, double a1, double phi, double omega, double ecc,
    double tsamp) {

    TimeDomainResampler resampler;
    // Interpolator resampler part.
    double minele, maxele;
    double binary_start_time = phi/n;
    thrust::host_vector<double> t_binary_grid(size);
    //Create a uniform grid every tsamp seconds from binary_start_time
    thrust::sequence(t_binary_grid.begin(), t_binary_grid.end(), binary_start_time, tsamp);
    thrust::device_vector<double> d_t_binary_grid = t_binary_grid;
    // t_telescope_nonuniform is the arrival time of the signal at the telescope after subtracting roemer delay. This is a non-uniform grid.
    thrust::host_vector<double> t_telescope_nonuniform(size);
    thrust::device_vector<double> d_t_telescope_nonuniform = t_telescope_nonuniform;
    /* Thrust vectors cannot be directly passed onto cuda kernels. Hence you need to cast them as raw pointers */
    double* d_t_binary_grid_ptr = thrust::raw_pointer_cast(d_t_binary_grid.data());
    double* d_t_telescope_nonuniform_ptr = thrust::raw_pointer_cast(d_t_telescope_nonuniform.data());
    resampler.subtract_roemer_delay_elliptical_bt_model(d_t_binary_grid_ptr, d_t_telescope_nonuniform_ptr, n, a1, phi, omega, ecc, tsamp, size);

    // Find min. and max. of the telescope arrival time non-uniform grid. We only use the min.
    find_min_max(d_t_telescope_nonuniform, &minele, &maxele);
    if (args.verbose) std::cout << "Roemer delay min: " << minele << ", max: " << maxele << "\n";
    /* Using minimum value of roemer delay, now generate your output samples.
    say minimum is 5400.0, array is then 5400, 5400 + tsamp, 5400 + 2*.tsamp + ... 5400 + (total_samples - 1) * tsamp */ 

    thrust::host_vector<double> t_binary_target(size);
    // Using the minima of d_t_telescope_nonuniform, our goal now is to create a target uniform grid separated every tsamp seconds.
    thrust::sequence(t_binary_target.begin(), t_binary_target.end(), minele, tsamp);
    thrust::device_vector<double> d_t_binary_target = t_binary_target;
    double* d_t_binary_target_ptr = thrust::raw_pointer_cast(d_t_binary_target.data());
    resampler.resample_using_1D_lerp(d_t_telescope_nonuniform_ptr, d_tim, size, d_t_binary_target_ptr, d_tim_resampled);
}

void exact_resampler_circular(
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled, unsigned int size,
    double n, double a1, double phi,
    double tsamp) {

    // Only the roemer delay part is different for circular orbit.

    TimeDomainResampler resampler;
    // Interpolator resampler part.
    double minele, maxele;
    double binary_start_time = phi/n;
    thrust::host_vector<double> t_binary_grid(size);
    //Create a uniform grid every tsamp seconds from binary_start_time
    thrust::sequence(t_binary_grid.begin(), t_binary_grid.end(), binary_start_time, tsamp);
    thrust::device_vector<double> d_t_binary_grid = t_binary_grid;
    // t_telescope_nonuniform is the arrival time of the signal at the telescope after subtracting roemer delay. This is a non-uniform grid.
    thrust::host_vector<double> t_telescope_nonuniform(size);
    thrust::device_vector<double> d_t_telescope_nonuniform = t_telescope_nonuniform;
    /* Thrust vectors cannot be directly passed onto cuda kernels. Hence you need to cast them as raw pointers */
    double* d_t_binary_grid_ptr = thrust::raw_pointer_cast(d_t_binary_grid.data());
    double* d_t_telescope_nonuniform_ptr = thrust::raw_pointer_cast(d_t_telescope_nonuniform.data());
    resampler.subtract_roemer_delay_circular(d_t_binary_grid_ptr, d_t_telescope_nonuniform_ptr, n, a1, phi, tsamp, size);
    // Find min. and max. of the telescope arrival time non-uniform grid. We only use the min.
    find_min_max(d_t_telescope_nonuniform, &minele, &maxele);
    if (args.verbose) std::cout << "Roemer delay min: " << minele << ", max: " << maxele << "\n";
    /* Using minimum value of roemer delay, now generate your output samples.
    say minimum is 5400.0, array is then 5400, 5400 + tsamp, 5400 + 2*.tsamp + ... 5400 + (total_samples - 1) * tsamp */ 

    thrust::host_vector<double> t_binary_target(size);
    // Using the minima of d_t_telescope_nonuniform, our goal now is to create a target uniform grid separated every tsamp seconds.
    thrust::sequence(t_binary_target.begin(), t_binary_target.end(), minele, tsamp);
    thrust::device_vector<double> d_t_binary_target = t_binary_target;
    double* d_t_binary_target_ptr = thrust::raw_pointer_cast(d_t_binary_target.data());
    resampler.resample_using_1D_lerp(d_t_telescope_nonuniform_ptr, d_tim, size, d_t_binary_target_ptr, d_tim_resampled);
}


       
void run_keplerian_search(int idx,
    Keplerian_TemplateBank_Reader& keplerian_tb,
    CandidateCollection& keplerian_search_cands,
    DedispersedTimeSeries<DedispOutputType>& tim,
    ReusableDeviceTimeSeries<float, DedispOutputType>& d_tim,
    DeviceTimeSeries<float>& d_tim_resampled,
    CuFFTerR2C& r2cfft,
    CuFFTerC2R& c2rfft,
    DeviceFourierSeries<cufftComplex>& d_fseries,
    DevicePowerSpectrum<float>& d_pspec,
    SpectrumFormer& former,
    HarmonicSums<float>& sums,
    HarmonicFolder& harm_folder,
    PeakFinder& cand_finder,
    HarmonicDistiller& harm_finder,
    float mean, float std, unsigned int size,
    double tsamp, double inverse_tsamp
    ) {

    const auto& n   = keplerian_tb.get_n();
    const auto& a1  = keplerian_tb.get_a1();
    const auto& phi = keplerian_tb.get_phi();
    const auto& omega  = keplerian_tb.get_omega();
    const auto& ecc  = keplerian_tb.get_ecc();


    if (elliptical_orbit_search) {
        if (args.exact_resampler) {
                if (args.verbose) std::cout << "Using exact BT model LERP resampler for elliptical orbit search\n";
                for (size_t kk = 0; kk < a1.size(); ++kk) {
                    exact_resampler_elliptical(d_tim, d_tim_resampled, size, n[kk], a1[kk], phi[kk], omega[kk], ecc[kk], tsamp);
                    SearchParams elliptical_orbit_search;
                    elliptical_orbit_search.n = n[kk];
                    elliptical_orbit_search.a1 = a1[kk];
                    elliptical_orbit_search.phi = phi[kk];
                    elliptical_orbit_search.omega = omega[kk];
                    elliptical_orbit_search.ecc = ecc[kk];
                    SpectrumCandidates trial_cands(tim.get_dm(), idx, elliptical_orbit_search);
                    run_search_and_find_candidates(
                        d_tim_resampled, r2cfft, c2rfft, d_fseries, d_pspec, former,
                        sums, harm_folder, cand_finder, harm_finder, trial_cands,
                        keplerian_search_cands, mean, std, size); 
                    // Update progress bar
                    if (prog) prog->tick_bin();     
            }
        }
        else {
            if (args.verbose) std::cout << "Using nearest neighbour resampler for elliptical orbit search\n";
            for (size_t kk = 0; kk < a1.size(); ++kk) {
                run_elliptical_orbit_search_resampler(d_tim, d_tim_resampled, size, n[kk], a1[kk], phi[kk], omega[kk], ecc[kk], tsamp, inverse_tsamp);
                SearchParams elliptical_orbit_search;
                elliptical_orbit_search.n = n[kk];
                elliptical_orbit_search.a1 = a1[kk];
                elliptical_orbit_search.phi = phi[kk];
                elliptical_orbit_search.omega = omega[kk];
                elliptical_orbit_search.ecc = ecc[kk];
                SpectrumCandidates trial_cands(tim.get_dm(), idx, elliptical_orbit_search);
                run_search_and_find_candidates(
                    d_tim_resampled, r2cfft, c2rfft, d_fseries, d_pspec, former,
                    sums, harm_folder, cand_finder, harm_finder, trial_cands,
                    keplerian_search_cands, mean, std, size);
                // Update progress bar
                if (prog) prog->tick_bin(); 
            }
        }

    }
    // Circular orbit search
    else {
        if (args.exact_resampler) {
            for (size_t kk = 0; kk < a1.size(); ++kk) {
                if (args.verbose) std::cout << "Using exact LERP resampler for circular orbit search\n";
                exact_resampler_circular(d_tim, d_tim_resampled, size, n[kk], a1[kk], phi[kk], tsamp);
                SearchParams circular_orbit_search;
                circular_orbit_search.n = n[kk];
                circular_orbit_search.a1 = a1[kk];
                circular_orbit_search.phi = phi[kk];
                SpectrumCandidates trial_cands(tim.get_dm(), idx, circular_orbit_search);
                run_search_and_find_candidates(
                    d_tim_resampled, r2cfft, c2rfft, d_fseries, d_pspec, former,
                    sums, harm_folder, cand_finder, harm_finder, trial_cands,
                    keplerian_search_cands, mean, std, size);
                // Update progress bar
                if (prog) prog->tick_bin();
            }
        }
        else {
            for (size_t kk = 0; kk < a1.size(); ++kk) {
                if (args.verbose) std::cout << "Using nearest neighbour resampler for circular orbit search\n";
                run_circular_orbit_search_resampler(d_tim, d_tim_resampled, size,  n[kk], a1[kk], phi[kk], tsamp, inverse_tsamp);
                SearchParams circular_orbit_search;
                circular_orbit_search.n = n[kk];
                circular_orbit_search.a1 = a1[kk];
                circular_orbit_search.phi = phi[kk];
                SpectrumCandidates trial_cands(tim.get_dm(), idx, circular_orbit_search);
                run_search_and_find_candidates(
                    d_tim_resampled, r2cfft, c2rfft, d_fseries, d_pspec, former,
                    sums, harm_folder, cand_finder, harm_finder, trial_cands,
                    keplerian_search_cands, mean, std, size);
                // Update progress bar
                if (prog) prog->tick_bin();
            }
        }
    }
}

public:
  CandidateCollection dm_trial_cands;
  std::vector<std::vector<Candidate>> window_trial_cands; // per window

  inline void find_min_max(thrust::device_vector<double> &dev_vec, double *min, double *max){
    thrust::pair<thrust::device_vector<double>::iterator,thrust::device_vector<double>::iterator> tuple;
    tuple = thrust::minmax_element(dev_vec.begin(),dev_vec.end());
    *min = *(tuple.first);
    *max = *tuple.second;
}

  Worker(DispersionTrials<DedispOutputType>& trials, DMDispenser& manager,
	 AccelerationPlan& acc_plan, CmdLineOptions& args, unsigned int size, int device, ProgressMonitor* prog,
         std::optional<Keplerian_TemplateBank_Reader>& keplerian_tb, bool elliptical_orbit_search,
         const std::vector<std::size_t>& window_starts, unsigned int window_size,
         AccelerationPlan* window_acc_plan, std::size_t dm_offset)
    :trials(trials)
    ,manager(manager)
    ,acc_plan(acc_plan)
    ,args(args)
    ,size(size)
    ,device(device)
    ,prog(prog)
    ,keplerian_tb(keplerian_tb)
    ,elliptical_orbit_search(elliptical_orbit_search)  
    ,window_starts(window_starts)
    ,window_size(window_size)
    ,window_acc_plan(window_acc_plan)
    ,dm_offset(dm_offset)
  {}

  void start(void)
  {
    
    cudaSetDevice(device);
    Stopwatch pass_timer;
    pass_timer.start();

    double tsamp = trials.get_tsamp();
    double inverse_tsamp = 1.0 / tsamp;
    DedispersedTimeSeries<DedispOutputType> tim, view;
    int idx;

    if(args.single_precision_harmonic_sums){
        if (args.verbose) std::cout << "Using single precision harmonic sums\n";
    }else {
        if (args.verbose) std::cout << "Using double precision harmonic sums\n";
    }
    
    prepare();
    window_trial_cands.assign(window_starts.size(), std::vector<Candidate>());
    std::vector<float> acc_list;

	PUSH_NVTX_RANGE("DM-Loop",0)
    while (true) {
        idx = manager.get_dm_trial_idx();
        trials.get_idx(idx, tim, size);
        if (idx == -1) break;
        int dm_idx = idx + (int)dm_offset;
        for (std::unique_ptr<Scale>& sp : scales) {
        Scale& s = *sp;
        const bool full = s.window < 0;
        DedispersedTimeSeries<DedispOutputType>& in = full ? tim : view;
        if (!full)
            scale_view(s, tim, view);
        //Start processing
        preprocess_time_series(in, s.d_tim, s.size);
        remove_rednoise_and_zap(s.d_tim, s.r2cfft, s.c2rfft, s.d_fseries, s.d_pspec, s.rednoise, bzap.get(),
                                s.former, s.mean, s.rms, s.std, s.size);

        if (keplerian_tb) {
            
            if (args.verbose) std::cout << "Searching Keplerian templates for DM " << tim.get_dm() << "\n";
            

            CandidateCollection keplerian_search_cands;
           
            run_keplerian_search(
                dm_idx, *keplerian_tb, keplerian_search_cands, tim, s.d_tim, s.d_tim_resampled,
                s.r2cfft, s.c2rfft, s.d_fseries, s.d_pspec, s.former, s.sums, s.harm_folder,
                s.cand_finder, s.harm_finder, s.mean, s.std, s.size, tsamp, inverse_tsamp);
            
            if (args.distill_circular_orbit_cands) {
                if (args.verbose) std::cout << "Distilling circular orbit candidates\n";
                Template_Bank_Circular_Distiller circular_distiller(args.freq_tol,true);
                dm_trial_cands.append(circular_distiller.distill(keplerian_search_cands.cands));
            } else {
                if (args.verbose) std::cout << "Not distilling circular orbit candidates\n";
                dm_trial_cands.append(keplerian_search_cands.cands);
            }
        }

        else {
        
            CandidateCollection accel_search_cands;
            
            // Acceleration search only
            run_acceleration_search(
                dm_idx, s.acc_plan, accel_search_cands, acc_list, in, s.d_tim, s.d_tim_resampled, s.r2cfft, s.c2rfft,
                s.d_fseries, s.d_pspec, s.former, s.sums, s.harm_folder, s.cand_finder,
                s.harm_finder, s.mean, s.std, s.size, full);

            if (args.verbose) std::cout << "Distilling accelerations" << std::endl;
            if (full) {
                dm_trial_cands.append(s.acc_still.distill(accel_search_cands.cands));
            } else {
                std::vector<Candidate> kept = s.acc_still.distill(accel_search_cands.cands);
                for (Candidate& c : kept)
                    c.window = s.window;
                std::vector<Candidate>& out = window_trial_cands[s.window];
                out.insert(out.end(), kept.begin(), kept.end());
            }
        }
        }
    }
	POP_NVTX_RANGE

    if (args.verbose)
      std::cout << "DM processing took " << pass_timer.getTime() << " seconds"<< std::endl;
  }

  // The light curves of CANDS, all of DM trial IDX: in every scale, the
  // largest value of each one's harmonic sum within one Fourier bin of its
  // frequency, after resampling to its acceleration (full_snr, window_snr).
  void light_curves(std::size_t idx, std::vector<Candidate*>& cands){
    cudaSetDevice(device);
    prepare();
    DedispersedTimeSeries<DedispOutputType> tim, view;
    trials.get_idx(idx, tim, size);
    std::map<float, std::vector<Candidate*>> by_acc;
    for (Candidate* c : cands) {
      c->window_snr.assign(window_starts.size(), NAN);
      by_acc[c->acc].push_back(c);
    }
    std::vector<float> values;
    for (std::unique_ptr<Scale>& sp : scales) {
      Scale& s = *sp;
      const bool full = s.window < 0;
      DedispersedTimeSeries<DedispOutputType>& in = full ? tim : view;
      if (!full)
        scale_view(s, tim, view);
      preprocess_time_series(in, s.d_tim, s.size);
      remove_rednoise_and_zap(s.d_tim, s.r2cfft, s.c2rfft, s.d_fseries, s.d_pspec, s.rednoise, bzap.get(),
                              s.former, s.mean, s.rms, s.std, s.size);
      for (auto& entry : by_acc) {
        run_acceleration_time_domain_resampler(s.d_tim, s.d_tim_resampled, entry.first, s.size);
        s.r2cfft.execute(s.d_tim_resampled.get_data(), s.d_fseries.get_data());
        s.former.form_interpolated(s.d_fseries, s.d_pspec);
        stats::normalise(s.d_pspec.get_data(), s.mean * s.size, s.std * s.size, s.size / 2 + 1);
        s.harm_folder.fold(s.d_pspec);
        for (Candidate* c : entry.second) {
          if (c->nh < 0 || c->nh > (int)s.sums.size())
            continue;
          DevicePowerSpectrum<float>& level = c->nh == 0 ? s.d_pspec : *s.sums[c->nh - 1];
          const long nbins = level.get_nbins();
          const long half = 1L << c->nh; // one Fourier bin of this scale
          const long centre = std::lround(c->freq / level.get_bin_width() * (double)half);
          const long lo = std::max(0L, centre - half);
          const long hi = std::min(nbins - 1, centre + half);
          if (lo > hi)
            continue;
          values.resize(hi - lo + 1);
          Utils::d2hcpy(values.data(), level.get_data() + lo, (unsigned int)values.size());
          const float best = *std::max_element(values.begin(), values.end());
          if (full)
            c->full_snr = best;
          else
            c->window_snr[s.window] = best;
        }
      }
    }
  }

};

void* launch_worker_thread(void* ptr){
  reinterpret_cast<Worker*>(ptr)->start();
  return NULL;
}


bool getFileContent(std::string fileName, std::vector<float> & vecOfDMs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    std::string str;
    float fl;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            fl = std::atof(str.c_str());
            //fl = std::stof(str); //c++11
            vecOfDMs.push_back(fl);
    }
    //Close The File
    in.close();
    return true;
}






// --windows N:HOP
static void parse_windows(std::string const& spec, std::size_t& nsamps, std::size_t& hop){
  std::size_t colon = spec.find(':');
  try {
    if (colon == std::string::npos)
      throw std::invalid_argument(spec);
    nsamps = std::stoull(spec.substr(0, colon));
    hop = std::stoull(spec.substr(colon + 1));
  } catch (std::exception&) {
    ErrorChecker::throw_error("--windows takes N:HOP, two sample counts");
  }
  if (nsamps < 2 || (nsamps & (nsamps - 1)) != 0 || hop == 0)
    ErrorChecker::throw_error("--windows needs N a power of two and HOP > 0");
}

// The DM trials (lowest, highest index) of a candidate's own signal: itself
// and its associates at its frequency (the DM distiller's), not the harmonics
// the harmonic distiller attached.
static std::pair<int,int> dm_idx_range(Candidate& c, float freq_tol){
  std::vector<CandidatePOD> all;
  c.collect_candidates(all);
  int lo = c.dm_idx, hi = c.dm_idx;
  for (CandidatePOD const& p : all) {
    if (std::fabs(p.freq / c.freq - 1.0) <= 2.0 * freq_tol) {
      lo = std::min(lo, p.dm_idx);
      hi = std::max(hi, p.dm_idx);
    }
  }
  return {lo, hi};
}

// Window candidates of one signal: within MERGE_BINS Fourier bins of a
// window (TOL Hz) and over neighbouring DM trials (gaps of up to
// MERGE_DM_GAP trials). The strongest is reported, with every window.
static const double MERGE_BINS = 2.0;
static const int MERGE_DM_GAP = 2;

static bool same_signal(Candidate& a, std::pair<int,int> ra, Candidate& b, std::pair<int,int> rb, double tol){
  return std::fabs(a.freq - b.freq) <= tol
      && ra.first <= rb.second + MERGE_DM_GAP && rb.first <= ra.second + MERGE_DM_GAP;
}

static std::vector<Candidate> merge_windows(std::vector<Candidate>& cands, double tol, float freq_tol){
  std::stable_sort(cands.begin(), cands.end(), [](Candidate const& a, Candidate const& b){ return a.snr > b.snr; });
  std::vector<Candidate> groups;
  std::vector<std::pair<int,int>> ranges;
  for (Candidate& c : cands) {
    std::pair<int,int> rc = dm_idx_range(c, freq_tol);
    std::size_t g = 0;
    while (g < groups.size() && !same_signal(groups[g], ranges[g], c, rc, tol))
      ++g;
    if (g == groups.size()) {
      groups.push_back(c);
      ranges.push_back(rc);
      continue;
    }
    std::vector<int>& w = groups[g].windows;
    for (int x : c.windows)
      if (std::find(w.begin(), w.end(), x) == w.end())
        w.push_back(x);
    ranges[g] = {std::min(ranges[g].first, rc.first), std::max(ranges[g].second, rc.second)};
  }
  for (Candidate& c : groups)
    std::sort(c.windows.begin(), c.windows.end());
  return groups;
}

int main(int argc, char **argv)
{
  std::map<std::string,Stopwatch> timers;
  timers["reading"]      = Stopwatch();
  timers["dedispersion"] = Stopwatch();
  timers["searching"]    = Stopwatch();
  timers["folding"]      = Stopwatch();
  timers["total"]        = Stopwatch();
  timers["total"].start();

  CmdLineOptions args;
  ProgressMonitor* progPtr = nullptr;
  if (!read_cmdline_options(args,argc,argv))
    ErrorChecker::throw_error("Failed to parse command line arguments.");

  int nthreads = std::min(Utils::gpu_count(),args.max_num_threads);
  nthreads = std::max(1,nthreads);

  /* Could do a check on the GPU memory usage here */

  if (args.verbose)
    std::cout << "Using file: " << args.infilename << std::endl;
  std::string filename(args.infilename);
  std::filesystem::path filpath = filename;

  if(args.timeseries_dump_dir == "" && args.no_search){
    std::cout << "-nosearch is only useful if you are only dumping timeseries. Otherwise it does nothing." << std::endl;
  }

 
  if (args.nsamples > 0 && args.size > 0 && args.nsamples > args.size) ErrorChecker::throw_error("nsamples cannot be > fft size.");
  const bool windowed = !args.windows.empty();
  std::size_t window_size = 0, window_hop = 0;
  std::vector<std::size_t> window_starts; // set from the first gulp's series length
  if (windowed) {
    parse_windows(args.windows, window_size, window_hop);
    if (args.keplerian_tb_file != "none")
      ErrorChecker::throw_error("--windows does not work with a Keplerian template bank");
    if (args.timeseries_dump_dir != "")
      ErrorChecker::throw_error("--windows does not work with --timeseries_dump_dir");
  }
  if (args.size > 0 && args.nsamples == 0){
     args.nsamples =  args.size;
    }  

  timers["reading"].start();
  SigprocFilterbank filobj(filename, args.start_sample, args.nsamples);
  SigprocHeader header = filobj.get_header();
  timers["reading"].stop();

  if (args.progress_bar){
    printf("Filterbank Reading Complete\n");
  }
  unsigned int size;
  if (args.size == 0){
    size =  Utils::prev_power_of_two(filobj.get_effective_nsamps()); // By this time  fft size = effective nsamps in the default case. 
  }
    else {
    size = args.size;
  }
  //Set for pepoch calculation later
  filobj.size = size;


  if (args.verbose)
    std::cout << "Effective nsamples " << filobj.get_effective_nsamps() << " points" << std::endl;


  DMDistiller dm_still(args.freq_tol,true);
  HarmonicDistiller harm_still(args.freq_tol,args.max_harm,true,false);
  CandidateCollection dm_cands;



  AccelerationPlan acc_plan(
    args.acc_start, // m/s^2
    args.acc_end,   // m/s^2
    args.acc_tol,   // dimensionless
    args.acc_pulse_width * 1e-6, // cmd line arg is microseconds but needs to be passed as seconds
    size, // Number of samples in FFT. Set based on segment samples and power of 2.
    filobj.get_tsamp(), // seconds
    filobj.get_cfreq() * 1e6, // from header in MHz needs converted to Hz
    filobj.get_foff() * 1e6 // from header in MHz needs converted to Hz
    );
 

  // The windows' own acceleration plan, for their FFT length
  AccelerationPlan window_acc_plan(
    args.acc_start, args.acc_end, args.acc_tol, args.acc_pulse_width * 1e-6,
    windowed ? window_size : size, filobj.get_tsamp(), filobj.get_cfreq() * 1e6, filobj.get_foff() * 1e6);
  std::vector<std::vector<Candidate>> window_cands;

  std::optional<Keplerian_TemplateBank_Reader> keplerian_tb;

  if (args.keplerian_tb_file != "none") {
    if (args.verbose)
        std::cout << "Using template bank file: " << args.keplerian_tb_file << std::endl;

    keplerian_tb.emplace(args.keplerian_tb_file);
    if (args.verbose) std::cout << "Loaded " << keplerian_tb->get_n().size()  << " templates\n";
}

  // Compute the “elliptical or circular” flag once
  bool elliptical_orbit_search = false;
  if (keplerian_tb && keplerian_tb->get_num_columns() == 5) {
    elliptical_orbit_search = true;
  }


  if (args.verbose)
    std::cout << "Generating DM list" << std::endl;
  std::vector<float> full_dm_list;
  std::optional<SeriesFile> series;

  if (args.series_file != "") {
    // The series were dedispersed already (heimdall -series_file)
    series.emplace(args.series_file);
    full_dm_list = series->get_dm_list();
    if (std::fabs(series->get_tsamp() - filobj.get_tsamp()) > 1e-6 * filobj.get_tsamp())
      ErrorChecker::throw_error("The series file's tsamp differs from the filterbank's");
    std::cout << "Reading " << full_dm_list.size() << " dedispersed series of "
              << series->get_nsamps() << " samples in " << series->get_block_count()
              << " blocks from " << args.series_file << std::endl;
  }
  else if (args.dm_file=="none") {
    Dedisperser dedisperser(filobj, nthreads);
    dedisperser.generate_dm_list(args.dm_start, args.dm_end, args.dm_pulse_width, args.dm_tol);
    full_dm_list = dedisperser.get_dm_list();
  }
  else {
      bool result = getFileContent(args.dm_file, full_dm_list);
  }

  float nbytes = args.host_ram_limit_gb * 1e9;
  std::size_t ndm_trial_gulp = std::size_t(nbytes / (filobj.get_effective_nsamps() * sizeof(float)));
  if (ndm_trial_gulp == 0)
  {
    throw std::runtime_error("Insufficient RAM specified to allow for dedispersion");
  }
  else if (ndm_trial_gulp > full_dm_list.size())
  {
    ndm_trial_gulp = full_dm_list.size();
  }
  for(std::size_t idx=0; idx < full_dm_list.size(); idx += ndm_trial_gulp){
    std::size_t start = idx;
    std::size_t end   = (idx + ndm_trial_gulp) > full_dm_list.size() ? full_dm_list.size(): (idx + ndm_trial_gulp) ;
    if(args.verbose) std::cout << "Gulp start: " << start << " end: " << end << std::endl;
    std::vector<float> dm_list_chunk(full_dm_list.begin() + start,  full_dm_list.begin() + end);
    std::optional<Dedisperser> dedisperser;
    if (!series) {
    dedisperser.emplace(filobj, nthreads);
    if (args.killfilename!=""){
      if (args.verbose)
        std::cout << "Using killfile: " << args.killfilename << std::endl;
      dedisperser->set_killmask(args.killfilename);
    }

    dedisperser->set_dm_list(dm_list_chunk);

    if (args.verbose){
    std::cout << dm_list_chunk.size() << " DM trials" << std::endl;
    for (std::size_t ii = 0; ii < dm_list_chunk.size(); ii++)
    {
      std::cout << dm_list_chunk[ii] << std::endl;
    }
    std::cout << "Executing dedispersion" << std::endl;
    }
    }

    timers["dedispersion"].start();
    PUSH_NVTX_RANGE("Dedisperse",3)
    DispersionTrials<DedispOutputType> trials(filobj.get_tsamp());
    if (series) {
    std::cout <<"loading dedispersed series...." <<std::endl;
    series->load(trials, start, end);
    } else {
    std::cout <<"dedispersing...." <<std::endl;

    std::size_t gulp_size;
    if (args.dedisp_gulp == -1){
      gulp_size = filobj.get_effective_nsamps();
    } else {
      gulp_size = args.dedisp_gulp;
    }
    if(args.verbose){
      std::cout<< "Starting to dedisperse filterbank from Sample:" << filobj.get_start_sample() 
            << " to " << filobj.get_end_sample() << " samples, with gulp size of " << gulp_size << " samples" << std::endl;
    }
    dedisperser->dedisperse(trials, filobj.get_start_sample(), filobj.get_end_sample(), gulp_size);
    }
    POP_NVTX_RANGE
    //trials.set_nsamps(size);
    timers["dedispersion"].stop();
    
    std::cout << "Setting FFT length to " << size << " points" << std::endl;
    if (windowed && window_cands.empty()) {
      for (std::size_t w0 = 0; w0 + window_size <= trials.get_nsamps(); w0 += window_hop)
        window_starts.push_back(w0);
      if (window_starts.empty())
        ErrorChecker::throw_error("--windows: a window is longer than the series");
      window_cands.resize(window_starts.size());
      std::cout << "Searching " << window_starts.size() << " windows of " << window_size
                << " samples every " << window_hop << " samples as well" << std::endl;
    }
    std::cout <<"Starting searching..."  << std::endl;

    //Multithreading commands
    timers["searching"].start();
    
    std::vector<Worker*> workers(nthreads);
    std::vector<pthread_t> threads(nthreads);

    std::size_t total_bin_trials;
    if (keplerian_tb) {
        total_bin_trials = full_dm_list.size() * keplerian_tb->get_n().size();
    } else {
        // number of acceleration trials changes with each DM.
        total_bin_trials = 0;
        std::vector<float> acc_tmp;
        for (float dm : full_dm_list) {
            acc_tmp.clear();
            acc_plan.generate_accel_list(dm, args.cdm, acc_tmp);
            total_bin_trials += acc_tmp.size();
        }
    }
   
    // choose your time step for the progress bar
    if (args.progress_bar) {
        double dm_step  = 0.05;
        double bin_step = 0.05;
        auto   t_step   = std::chrono::seconds(3600); // 1 hour
        progPtr = new ProgressMonitor(full_dm_list.size(),
                                    total_bin_trials,
                                    dm_step, bin_step, t_step);
    }

    DMDispenser dispenser(trials, progPtr);
 

    if (args.timeseries_dump_dir != ""){
      std::cout << "Dumping time series to " << args.timeseries_dump_dir << std::endl;
      std::cout << "filename without ext: " << filpath.stem().string() << std::endl;
      std::filesystem::create_directories(args.timeseries_dump_dir);
      for (int ii=0;ii<dm_list_chunk.size();ii++){
        trials.write_timeseries_to_file(args.timeseries_dump_dir, filpath.stem().string(), ii, header);
      }
      if(args.no_search){
        std::cout << "No search requested, exiting" << std::endl;
        return 0;
      }
    }
    
    for (int ii=0;ii<nthreads;ii++){
      workers[ii] = (new Worker(trials,dispenser,acc_plan,args,size,ii,progPtr,keplerian_tb,elliptical_orbit_search,
                                window_starts,window_size,&window_acc_plan,start));
      pthread_create(&threads[ii], NULL, launch_worker_thread, (void*) workers[ii]);
    }

    if(args.verbose)
      std::cout << "Joining worker threads" << std::endl;

    for (int ii=0; ii<nthreads; ii++){
      pthread_join(threads[ii],NULL);
      dm_cands.append(workers[ii]->dm_trial_cands.cands);
      for (std::size_t w = 0; w < window_cands.size(); ++w)
        window_cands[w].insert(window_cands[w].end(), workers[ii]->window_trial_cands[w].begin(),
                               workers[ii]->window_trial_cands[w].end());
      delete workers[ii];
    }
    if (progPtr) progPtr->finish(); 
    timers["searching"].stop();

  }

  if (args.verbose)
    std::cout << "Distilling DMs" << std::endl;
  
  dm_cands.cands = dm_still.distill(dm_cands.cands);
  dm_cands.cands = harm_still.distill(dm_cands.cands);

  CandidateScorer cand_scorer(filobj.get_tsamp(),filobj.get_cfreq(), filobj.get_foff(),
			      fabs(filobj.get_foff())*filobj.get_nchans());
  cand_scorer.score_all(dm_cands.cands);

  if (args.verbose)
    std::cout << "Setting up time series folder" << std::endl;

  if (args.verbose)
    std::cout << "Writing output files" << std::endl;

  int new_size = std::min(args.limit,(int) dm_cands.cands.size());
  dm_cands.cands.resize(new_size);

  if (windowed) {
    // Each window as the whole series is, then one candidate per signal
    // across the windows
    timers["windows"] = Stopwatch();
    timers["windows"].start();
    std::vector<Candidate> window_found;
    const float wtol = window_freq_tol(args.freq_tol, size, window_size);
    for (std::size_t w = 0; w < window_cands.size(); ++w) {
      DMDistiller dm_still_w(wtol,true);
      HarmonicDistiller harm_still_w(wtol,args.max_harm,true,false);
      std::vector<Candidate> found = dm_still_w.distill(window_cands[w]);
      found = harm_still_w.distill(found);
      for (Candidate& c : found) {
        c.window = (int)w;
        c.windows.assign(1, (int)w);
        c.in_full = false;
      }
      window_found.insert(window_found.end(), found.begin(), found.end());
    }
    cand_scorer.score_all(window_found);
    const double tol = MERGE_BINS / (window_size * filobj.get_tsamp());
    std::size_t n_window = window_found.size();
    window_found = merge_windows(window_found, tol, wtol);
    // A window signal the whole series found too joins that candidate,
    // which is reported from the more significant of the two
    std::vector<std::pair<int,int>> full_ranges;
    for (Candidate& c : dm_cands.cands) {
      full_ranges.push_back(dm_idx_range(c, args.freq_tol));
      c.window = -1;
      c.in_full = true;
    }
    std::vector<Candidate> window_only;
    for (Candidate& wc : window_found) {
      std::pair<int,int> rw = dm_idx_range(wc, wtol);
      int best = -1;
      for (std::size_t i = 0; i < dm_cands.cands.size(); ++i)
        if (same_signal(dm_cands.cands[i], full_ranges[i], wc, rw, tol)
            && (best < 0 || dm_cands.cands[i].snr > dm_cands.cands[best].snr))
          best = (int)i;
      if (best < 0) {
        window_only.push_back(wc);
        continue;
      }
      Candidate& fc = dm_cands.cands[best];
      std::vector<int> windows = fc.windows;
      for (int x : wc.windows)
        if (std::find(windows.begin(), windows.end(), x) == windows.end())
          windows.push_back(x);
      std::sort(windows.begin(), windows.end());
      if (wc.snr > fc.snr)
        fc = wc;
      fc.windows = windows;
      fc.in_full = true;
    }
    std::stable_sort(dm_cands.cands.begin(), dm_cands.cands.end(),
                     [](Candidate const& a, Candidate const& b){ return a.snr > b.snr; });
    if ((int)window_only.size() > args.limit)
      window_only.resize(args.limit);
    std::cout << n_window << " window candidates are " << window_found.size() << " signals; "
              << window_found.size() - window_only.size() << " joined whole-series candidates, "
              << window_only.size() << " kept as window candidates" << std::endl;
    dm_cands.cands.insert(dm_cands.cands.end(), window_only.begin(), window_only.end());

    // Light curves, from the series file
    if (series && !args.no_light_curves) {
      std::map<int, std::vector<Candidate*>> by_dm;
      for (Candidate& c : dm_cands.cands)
        by_dm[c.dm_idx].push_back(&c);
      DispersionTrials<DedispOutputType> lc_trials(filobj.get_tsamp());
      DMDispenser lc_dispenser(lc_trials);
      Worker lc_worker(lc_trials, lc_dispenser, acc_plan, args, size, 0, nullptr, keplerian_tb, false,
                       window_starts, window_size, &window_acc_plan, 0);
      for (auto& entry : by_dm) {
        series->load(lc_trials, entry.first, entry.first + 1);
        lc_worker.light_curves(0, entry.second);
      }
      std::cout << "Light curves of " << dm_cands.cands.size() << " candidates from " << by_dm.size()
                << " DM trials" << std::endl;
    } else if (!args.no_light_curves) {
      std::cout << "No light curves: they need --series_file" << std::endl;
    }
    timers["windows"].stop();
  }

  CandidateFileWriter cand_files(args.outdir);
  //cand_files.write_binary(dm_cands.cands,"candidates.peasoup");

  OutputFileWriter stats;
  stats.add_misc_info();
  stats.add_header(filename);
  if (keplerian_tb) {
    stats.add_search_parameters(args, &*keplerian_tb);
  } else {
    stats.add_search_parameters(args, nullptr);
  }
  stats.add_segment_parameters(filobj, args);
  stats.add_dm_list(full_dm_list);

  std::vector<float> acc_list;
  acc_plan.generate_accel_list(args.cdm, args.cdm, acc_list);
  stats.add_acc_list(acc_list, args.cdm);

  std::vector<int> device_idxs;
  for (int device_idx=0;device_idx<nthreads;device_idx++)
    device_idxs.push_back(device_idx);
  stats.add_gpu_info(device_idxs);
  if (windowed) {
    stats.add_windows(window_starts, window_size, window_hop);
    stats.add_candidates(dm_cands.cands,cand_files.byte_mapping,filobj,&window_starts,window_size,size);
  } else {
    stats.add_candidates(dm_cands.cands,cand_files.byte_mapping,filobj);
  }
  timers["total"].stop();
  stats.add_timing_info(timers);

  std::stringstream xml_filepath;
  xml_filepath << args.outdir << "/" << "overview.xml";
  stats.to_file(xml_filepath.str());

  if (progPtr) {
        delete progPtr;
        progPtr = nullptr;
    }
  //Print time for dedispersion, searching and total
  if (args.progress_bar) {
        std::cout << "Dedispersion took " << timers["dedispersion"].getTime() << " seconds" << std::endl;
        std::cout << "Searching took " << timers["searching"].getTime() << " seconds" << std::endl;
        std::cout << "Total time taken: " << timers["total"].getTime() << " seconds" << std::endl;
    }
  std::cout << "all done" << std::endl;

  return 0;
}
