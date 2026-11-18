#pragma once
#include <iostream>
#include <vector>
#include <sstream>
#include "stdio.h"
#include <cmath>


struct CandidatePOD {
  float dm;
  int dm_idx;
  float acc;
  float jerk;
  double n; //angular velocity = 2 * pi / orbital period
  double a1; //projected semi-major axis in light seconds
  double phi; //orbital phase
  double omega; //longitude of periastron
  double ecc; //eccentricity
  int nh;
  float snr;
  float freq;

};

struct Candidate {
public:
  float dm;
  int dm_idx;
  float acc;
  float jerk;
  double n; 
  double a1; 
  double phi; 
  double omega;
  double ecc;
  int nh;
  float snr;
  float freq;
  float folded_snr;
  double opt_period;
  bool is_adjacent;
  bool is_physical;
  float ddm_count_ratio;
  float ddm_snr_ratio;
  std::vector<Candidate> assoc;
  std::vector<float> fold;
  int nbins;
  int nints;
  // With --windows: the scale the candidate is reported from (-1 the whole
  // series, else a window index), the windows the signal was found in,
  // whether the whole-series search found it, and its light curve (S/N at
  // its frequency, harmonic sum, DM trial and acceleration in the whole
  // series and in each window; NaN where not measured).
  int window = -1;
  std::vector<int> windows;
  bool in_full = true;
  float full_snr = NAN;
  std::vector<float> window_snr;
  
  void append(Candidate& other){
    assoc.push_back(other);
  }

  int count_assoc(void){
    int count = 0;
    for (int ii=0;ii<assoc.size();ii++){
      count ++;
      count += assoc[ii].count_assoc();
    }
    return count;
  }
  
  Candidate(float dm, int dm_idx, float acc, float jerk, double n, double a1, double phi, double omega, double ecc, int nh, float snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),jerk(jerk),
     n(n),a1(a1),phi(phi),omega(omega),ecc(ecc),nh(nh),
     snr(snr),folded_snr(0.0),freq(freq),opt_period(0.0),
     is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}
  
  Candidate(float dm, int dm_idx, float acc, float jerk, double n, double a1, double phi, double omega, double ecc, int nh, float snr, float folded_snr, float freq)
    :dm(dm),dm_idx(dm_idx),acc(acc),jerk(jerk),n(n),a1(a1),
     phi(phi),omega(omega),ecc(ecc),nh(nh),snr(snr),
     folded_snr(folded_snr),freq(freq),opt_period(0.0),
     is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}

  Candidate()
    :dm(0.0),dm_idx(0.0),acc(0.0),jerk(0.0),n(0.0),a1(0.0),
     phi(0.0),omega(0.0),ecc(0.0),nh(0.0),snr(0.0),
     folded_snr(0.0),freq(0.0),opt_period(0.0),
     is_adjacent(false),is_physical(false),
     ddm_count_ratio(0.0),ddm_snr_ratio(0.0),nints(0),nbins(0){}

  void set_fold(float* ar, int nbins, int nints){
    int size = nbins*nints;
    this->nints = nints;
    this->nbins = nbins;
    fold.resize(size);
    for (int ii=0;ii<size;ii++)
      fold[ii] = ar[ii];
  }
  
  void collect_candidates(std::vector<CandidatePOD>& cands_lite){
    CandidatePOD cand_stats = {dm,dm_idx,acc,jerk,n,a1,phi,omega,ecc,nh,snr,freq};
    cands_lite.push_back(cand_stats);    
    for (int ii=0;ii<assoc.size();ii++){
      assoc[ii].collect_candidates(cands_lite);
    }
  }

  void print(FILE* fo=stdout){
    fprintf(fo,"%.15f\t%.15f\t%.15f\t%.2f\t%.4f\t%.4f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%d\t%.1f\t%.1f\t%d\t%d\t%.4f\t%.4f\t%zu\n",
	    1.0/freq,opt_period,freq,dm,acc,jerk,
        n,a1,phi,omega,ecc,nh,snr,folded_snr,is_adjacent,
	    is_physical,ddm_count_ratio,
	    ddm_snr_ratio,assoc.size());
    for (int ii=0;ii<assoc.size();ii++){
      assoc[ii].print(fo);
    }
  }

};

class CandidateCollection {
public:
  std::vector<Candidate> cands;

  CandidateCollection(){}
  
  void append(CandidateCollection& other){
    cands.insert(cands.end(),other.cands.begin(),other.cands.end());
  }
  
  void append(std::vector<Candidate> other){
    cands.insert(cands.end(),other.begin(),other.end());
  }

  void reset(void){
    cands.clear();
  }

  void print(FILE* fo=stdout){
    for (int ii=0;ii<cands.size();ii++)
      cands[ii].print(fo);
  }

  void generate_candidate_binaries(std::string output_directory="./") {
    char filename[80];    
    std::stringstream filepath;
    for (int ii=0;ii<cands.size();ii++){
      filepath.str("");
      sprintf(filename,"cand_%04d_%.5f_%.1f_%.1f.peasoup",
	      ii,1.0/cands[ii].freq,cands[ii].dm,cands[ii].acc);
      filepath << output_directory << "/" << filename;
      FILE* fo = fopen(filepath.str().c_str(),"w");
      
      cands[ii].print(fo);
      fclose(fo);
    }
  }
  
  void write_candidate_file(std::string filepath="./candidates.txt") {
    FILE* fo = fopen(filepath.c_str(),"w");
    fprintf(fo,"#Period...Opt_Period...Freq..DM..Acc..Jerk..N..A1..Phi..Omega..Ecc..NH..SNR..Folded_SNR..Is_Adjacent..Is_Physical..DDM_Count_Ratio..DDM_SNR_Ratio..Assoc_Count\n");
    for (int ii=0;ii<cands.size();ii++){
      fprintf(fo,"#Candidate %d\n",ii);
      cands[ii].print(fo);
    }
    fclose(fo);
  }
};

struct SearchParams {
  float acc        = 0.0f;
  float jerk       = 0.0f;
  double n         = 0.0;
  double a1        = 0.0;
  double phi       = 0.0;
  double omega     = 0.0;
  double ecc       = 0.0;
  int binary_idx   = -1;
};

class SpectrumCandidates : public CandidateCollection {
public:
  float dm;
  int dm_idx;
  float acc;
  float jerk;
  double n;
  double a1;
  double phi;
  double omega;
  double ecc;
  int binary_idx;

  SpectrumCandidates(float dm_, int dm_idx_, const SearchParams& opts = SearchParams())
    : dm(dm_), dm_idx(dm_idx_),
      acc(opts.acc), jerk(opts.jerk),
      n(opts.n), a1(opts.a1), phi(opts.phi),
      omega(opts.omega), ecc(opts.ecc),
      binary_idx(opts.binary_idx) {}

  void append(float* snrs, float* freqs, int nh, int size) {
    cands.reserve(cands.size() + size);
    for (int i = 0; i < size; i++) {
      cands.push_back(Candidate(dm, dm_idx, acc, jerk, n, a1, phi, omega, ecc,
                                nh, snrs[i], freqs[i]));
    }
  }
};
