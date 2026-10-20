#pragma once
#include <data_types/timeseries.hpp>
#include <utils/exceptions.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

/*!
  \brief Reader for the dedispersed series that heimdall writes with
  -series_file, so that a search can use them instead of dedispersing.

  The layout, all little-endian:
    char[8]  "HDSERIES"
    uint32   version (1)
    uint32   dm_count
    uint32   nchans
    uint32   nbits per sample (32: float)
    float64  dt (s), f0 (MHz), df (MHz)
    uint64   max_delay (samples)
    float32  dm_list[dm_count]
  then one block per heimdall gulp:
    uint64   first_idx (sample index of the block's first sample)
    uint64   nsamps
    float32  series[dm_count][nsamps]
  Blocks can overlap; the later block wins.
*/
class SeriesFile {
private:
  struct Block {
    std::uint64_t first_idx;
    std::uint64_t nsamps;
    std::uint64_t offset; // Of the block's data in the file
  };
  std::string filename;
  FILE* file;
  std::uint32_t nchans;
  double dt, f0, df;
  std::uint64_t max_delay;
  std::vector<float> dm_list;
  std::vector<Block> blocks;
  std::size_t nsamps;

  void read(void* data, std::size_t size){
    if (fread(data, 1, size, file) != size)
      ErrorChecker::throw_error("Could not read " + filename);
  }

public:
  /*!
    \brief Read the header and find the blocks of a series file.
  */
  SeriesFile(std::string filename)
    :filename(filename), nsamps(0)
  {
    file = fopen(filename.c_str(), "rb");
    if (!file)
      ErrorChecker::throw_error("Could not open " + filename);
    char magic[8];
    std::uint32_t ints[4];
    double doubles[3];
    read(magic, sizeof(magic));
    read(ints, sizeof(ints));
    if (std::memcmp(magic, "HDSERIES", 8) != 0 || ints[0] != 1 || ints[3] != 32)
      ErrorChecker::throw_error(filename + " is not a version 1, 32-bit series file");
    read(doubles, sizeof(doubles));
    read(&max_delay, sizeof(max_delay));
    nchans = ints[2];
    dt = doubles[0];
    f0 = doubles[1];
    df = doubles[2];
    dm_list.resize(ints[1]);
    read(dm_list.data(), dm_list.size() * sizeof(float));
    std::uint64_t header[2];
    while (fread(header, 1, sizeof(header), file) == sizeof(header)) {
      Block block = {header[0], header[1], (std::uint64_t)ftello(file)};
      blocks.push_back(block);
      nsamps = std::max<std::size_t>(nsamps, block.first_idx + block.nsamps);
      if (fseeko(file, block.nsamps * dm_list.size() * sizeof(float), SEEK_CUR) != 0)
        ErrorChecker::throw_error("Could not seek in " + filename);
    }
  }

  ~SeriesFile(){ fclose(file); }

  std::vector<float> const& get_dm_list(void){return dm_list;}
  double get_tsamp(void){return dt;}
  std::size_t get_nsamps(void){return nsamps;}
  std::size_t get_block_count(void){return blocks.size();}

  /*!
    \brief Load the series of DMs [dm_start, dm_end) into trials.
  */
  void load(DispersionTrials<float>& trials, std::size_t dm_start, std::size_t dm_end){
    std::vector<float> chunk(dm_list.begin() + dm_start, dm_list.begin() + dm_end);
    trials.resize(nsamps, chunk);
    float* ptr = trials.get_data_ptr();
    for (std::size_t b = 0; b < blocks.size(); ++b) {
      Block const& block = blocks[b];
      if (fseeko(file, block.offset + dm_start * block.nsamps * sizeof(float), SEEK_SET) != 0)
        ErrorChecker::throw_error("Could not seek in " + filename);
      for (std::size_t d = dm_start; d < dm_end; ++d)
        read(ptr + (d - dm_start) * nsamps + block.first_idx, block.nsamps * sizeof(float));
    }
  }
};
