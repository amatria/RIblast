/*
 * fastafile_reader.cpp
 *
 *  Created on: 2016/8/31
 *      Author: Tsukasa Fukunaga
 */

#include "fastafile_reader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "mpi.h"
#include "minmaxheap.h"

using namespace minmaxheap;

struct proc {
  double chars;
  int rank, r_chars;
  vector<int> indices;

  proc(int rank, double chars, int r_chars) {
    this->rank = rank;
    this->chars = chars;
    this->r_chars = r_chars;
  }

  bool operator <(const proc& x) const {
    return chars < x.chars;
  }

  bool operator >(const proc& x) const {
    return chars > x.chars;
  }
};

struct node {
  int idx, size;

  node(int idx, int size) {
      this->idx = idx;
      this->size = size;
  }

  bool operator <(const node& x) const {
    return size < x.size;
  }
};

const bool sort_procs(const proc& x, const proc& y) {
  return x.rank < y.rank;
}

void CountSequences(string input_file_name, vector<node> &nodes) {
  int count = 1;
  string buffer;
  ifstream fp;

  fp.open(input_file_name.c_str(), ios::in);
  if (!fp) {
    cout << "Error: can't open input_file: " << input_file_name << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  getline(fp, buffer);
  node n(count, 0);
  while (getline(fp, buffer)) {
    if (buffer[0] == '>') {
        nodes.push_back(n);
        n = node(++count, 0);
    } else {
      if (buffer.size() >= 2) {
        if (buffer.substr(buffer.size() - 2, 2) == "\r\n") {
          buffer.erase(buffer.size() - 2, 2);
        }
      }
      if (buffer[buffer.size() - 1] == '\r' || buffer[buffer.size() - 1] == '\n') {
        buffer.erase(buffer.size() - 1, 1);
      }
      n.size += buffer.size();
    }
  }
  nodes.push_back(n);
  fp.close();
}

void FastafileReader::ReadFastafile(string input_file_name, vector<string> &sequences, vector<string> &names){
  int rank, procs;
  vector<int> idx;
  string buffer;
  ifstream fp;
  int *counts, *indices, *displ;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  counts = (int *) malloc(procs * sizeof(int));
  if (rank == 0) {
    vector<node> nodes;
    MinMaxHeap<proc> proc_heap(procs);
    CountSequences(input_file_name, nodes);

    displ = (int *) malloc(procs * sizeof(int));
    indices = (int *) malloc(nodes.size() * sizeof(int));

    // populate min max heap
    for (int i = 0; i < procs; i++) {
      proc_heap.insert(proc(i, 0, 0));
    }

    sort(nodes.begin(), nodes.end());
    while (!nodes.empty()) {
      node n = nodes[0];
      proc p = proc_heap.popmin();

      p.r_chars += n.size;
      p.chars += pow(n.size, 1);
      p.indices.push_back(n.idx);

      proc_heap.insert(p);
      nodes.erase(nodes.begin());
    }

    int k = 0;
    vector<proc> proc_array = proc_heap.getheap();
    sort(proc_array.begin(), proc_array.end(), sort_procs);
    for (int i = 0; i < procs; i++) {
      proc p = proc_array[i];

      cout << "Process #" << p.rank << " received " << p.indices.size() << " sequences (" << p.r_chars << " chars).\n";
      counts[i] = p.indices.size();
      for (int j = 0; j < counts[i]; j++) {
        indices[k++] = p.indices[j];
      }

      displ[i] = (i == 0 ? 0 : displ[i - 1] + counts[i - 1]);
    }
    cout.flush();
  }

  MPI_Bcast(counts, procs, MPI_INT, 0, MPI_COMM_WORLD);
  idx.resize(counts[rank]);
  MPI_Scatterv(indices, counts, displ, MPI_INT, idx.data(), idx.size(),
               MPI_INT, 0, MPI_COMM_WORLD);

  free(counts);
  if (rank == 0) {
    free(indices);
    free(displ);
  }

  sort(idx.begin(), idx.end());
  fp.open(input_file_name.c_str(), ios::in);
  if (!fp) {
    cout << "Error: can't open input_file: " << input_file_name << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int t = 0;
  int i = 0;
  getline(fp, buffer);
  string tmp_seq = "";
  while (true) {
    if (i == idx.size() || fp.eof()) {
      break;
    }
    if (buffer[0] == '>' && ++t == idx[i]) {
      names.push_back(buffer.substr(1, buffer.size() - 1));
      while (getline(fp, buffer)) {
        if (buffer[0] == '>') {
          sequences.push_back(tmp_seq);
          tmp_seq = "";
          i++;
          break;
        } else {
          if (buffer.size() >= 2) {
            if (buffer.substr(buffer.size() - 2, 2) == "\r\n") {
              buffer.erase(buffer.size() - 2, 2);
            }
          }
          if (buffer[buffer.size() - 1] == '\r' || buffer[buffer.size() - 1] == '\n') {
            buffer.erase(buffer.size() - 1, 1);
          }
          tmp_seq = tmp_seq + buffer;
        }
      }
    } else {
      getline(fp, buffer);
    }
  }

  if (fp.eof()) {
    sequences.push_back(tmp_seq);
  }

  fp.close();
}

void FastafileReader::ReadFastafile(string input_file_name, vector<vector<string>> &vec_sequences, vector<vector<string>> &vec_names, int max_seqs) {
  ifstream fp;
  string buffer;

  vector<string> sequences;
  vector<string> names;

  fp.open(input_file_name.c_str(), ios::in);
  if (!fp){
    cout << "Error: can't open input_file: " << input_file_name << "." << endl;
    exit(1);
  }

  getline(fp, buffer);

  int count = 1;
  string temp_sequence = "";
  names.push_back(buffer.substr(1, buffer.size() - 1));

  while (getline(fp, buffer)) {
    if (buffer[0] == '>'){
      sequences.push_back(temp_sequence);
      if (count++ == max_seqs) {
        count = 0;

        vec_names.push_back(names);
        vec_sequences.push_back(sequences);

        names.clear();
        sequences.clear();
      }
      names.push_back(buffer.substr(1, buffer.size() - 1));
      temp_sequence = "";
    } else {
      if (buffer.size() >= 2) {
	      if (buffer.substr(buffer.size() - 2, 2) == "\r\n") {
	        buffer.erase(buffer.size() - 2, 2);
	      }
      }
      if (buffer[buffer.size() - 1] == '\r' || buffer[buffer.size() - 1] == '\n') {
	      buffer.erase(buffer.size() - 1, 1);
      }
      temp_sequence = temp_sequence + buffer;
    }
  }
  sequences.push_back(temp_sequence);

  vec_names.push_back(names);
  vec_sequences.push_back(sequences);

  fp.close();
}

void FastafileReader::ReadFastafile(string input_file_name, string &sequence, string &name){
  ifstream fp;
  string buffer;
  fp.open(input_file_name.c_str(),ios::in);
  if (!fp){
    cout << "Error: can't open input_file:"+input_file_name+"." <<endl;
    exit(1);
  }
  getline(fp,buffer);
  name = buffer.substr(1,buffer.size()-1);
  sequence = "";
  while(getline(fp,buffer)){
    if(buffer.size()>=2){
      if(buffer.substr(buffer.size()-2,2) == "\r\n"){
	buffer.erase(buffer.size()-2,2);
      }
    }
    if(buffer[buffer.size()-1] == '\r' || buffer[buffer.size()-1] == '\n'){
      buffer.erase(buffer.size()-1,1);
    }
    sequence = sequence + buffer;
  }
  fp.close();
}
