/*
 * rna_interaction_search.cpp
 *
 *     Created on: 2016/8/31
 *  Last modified: 2016/01/25
 *         Author: Tsukasa Fukunaga
 */

#include "rna_interaction_search.h"
#include "fastafile_reader.h"
#include "encoder.h"
#include "sais.h"
#include "raccess.h"
#include "seed_search.h"
#include "ungapped_extension.h"
#include "gapped_extension.h"
#include <fstream>
#include <math.h>
#include <algorithm>
#include "mpi.h"
#include <omp.h>
#include <sstream>
#include "database_reader.h"
#include <sys/stat.h>
#include <unordered_map>

struct acc_node {
  int idx;
  float val;

  acc_node(int idx, float val) {
    this->idx = idx;
    this->val = val;
  }

  bool operator <(const acc_node& x) const {
    return val < x.val;
  }
};

class sort_indices {
  private:
    vector<string> *_sequences;
  public:
    sort_indices(vector<string> *sequences) {
      _sequences = sequences;
    }
    bool operator()(int i, int j) const {
      return _sequences->at(j).size() < _sequences->at(i).size();
    }
};

class sort_acc {
  private:
    vector<int> *_idx;
  public:
    sort_acc(vector<int> *idx) {
      _idx = idx;
    }
    bool operator()(int i, int j) const {
      return _idx->at(i) < _idx->at(j);
    }
};

bool compare(const Hit& left, const Hit& right) {
  if(left.GetDbSp() != right.GetDbSp()){
    return(left.GetDbSp() < right.GetDbSp());
  }else if(left.GetQSp() != right.GetQSp()){
    return(left.GetQSp() < right.GetQSp());
  }else if(left.GetDbLength() != right.GetDbLength()){
    return(left.GetDbLength() > right.GetDbLength());
  }else{
    return(left.GetQLength() > right.GetQLength());
  }
}

bool base_pair_compare(const BasePair& left, const BasePair& right) {
  return(left.qpos < right.qpos);
}

string acc_dir(string path) {
  stringstream s;
  if (path != "") {
    s << path << "/";
  }
  s << "priblast_acc";
  return s.str();
}

string acc_file(string path, int idx) {
  stringstream s;
  if (path != "") {
    s << path << "/";
  }
  s << idx;
  return s.str();
}

string my_acc_file(string path, int rank, int thread) {
    stringstream s;
  if (path != "") {
    s << path << "/";
  }
  s << "_" << rank << "_" << thread;
  return s.str();
}

string my_output_file(int rank, int id, string path) {
  stringstream s;
  if (path != "") {
    s << path << "/";
  }
  s << "priblast_out_ " << rank << "_" << id;
  return s.str();
}

void create_output_file(string output_file) {
  ofstream ofs;
  ofs.open(output_file, ios::out);
  if (!ofs) {
    cout << "Error: can't open output_file: " << output_file << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ofs.close();
}

int merge_output(const RnaInteractionSearchParameters parameters, int rank,
                 int threads, int displ, int crt) {
  int count = displ;
  ifstream ifs;
  ofstream ofs;

  if (crt) {
    ofs.open(parameters.GetOutputFilename().c_str(), ios::out);
  } else {
    ofs.open(parameters.GetOutputFilename().c_str(), ios::app);
  }

  if (!ofs) {
    cout << "Error: can't open output_file: " << parameters.GetOutputFilename() << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (crt) {
    ofs << "RIblast ris result\n";
    ofs << "input:" << parameters.GetInputFilename() << ",database:" << parameters.GetDbFilename() << ",RepeatFlag:" << parameters.GetRepeatFlag() << ",MaximalSpan:" << parameters.GetMaximalSpan() << ",MinAccessibleLength:" << parameters.GetMinAccessibleLength() << ",MaxSeedLength:" << parameters.GetMaxSeedLength() << ",InteractionEnergyThreshold:" << parameters.GetInteractionEnergyThreshold() << ",HybridEnergyThreshold:" << parameters.GetHybridEnergyThreshold() << ",FinalThreshold:" << parameters.GetFinalThreshold() << ",DropOutLengthWoGap:" << parameters.GetDropOutLengthWoGap() << ",DropOutLengthWGap:" << parameters.GetDropOutLengthWGap() << "\n";
    ofs << "Id,Query name, Query Length, Target name, Target Length, Accessibility Energy, Hybridization Energy, Interaction Energy, BasePair\n";
  }

  for (int i = 0; i < threads; i++) {
    string input_file = my_output_file(rank, i, parameters.GetPath());

    ifs.open(input_file.c_str(), ios::in);
    if (!ifs) {
      cout << "Error: can't read input_file: " << input_file << "." << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string buffer;
    while (getline(ifs, buffer)) {
      ofs << count++ << "," << buffer << "\n";
    }

    ifs.close();
    if (remove(input_file.c_str()) != 0) {
      cout << "Warning: there was an error deleting input_file: " << input_file << "." << endl;
    }
  }

  ofs.close();
  return count;
}

float save_acc(string output_file, vector<float> &acc,
               vector<float> &cond_acc) {
  ofstream ofs;
  float f;
  int s;

  ofs.open(output_file.c_str(), ios::out | ios::binary);
  if (!ofs) {
    cout << "Error: can't open acc_file: " << output_file << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  s = acc.size();
  ofs.write(reinterpret_cast<const char*>(&s), sizeof(int));
  for (int i = 0; i < s; i++) {
    f = acc[i];
    ofs.write(reinterpret_cast<const char*>(&f), sizeof(float));
  }
  s = cond_acc.size();
  ofs.write(reinterpret_cast<const char*>(&s), sizeof(int));
  for (int i = 0; i < s; i++) {
    f = cond_acc[i];
    ofs.write(reinterpret_cast<const char*>(&f), sizeof(float));
  }
  ofs.close();

  f = 0;
  for (int i = 0; i < acc.size(); i++) {
    f += acc[i];
  }
  for (int i = 0; i < cond_acc.size(); i++) {
    f += cond_acc[i];
  }
  return f;
}

void load_acc(string input_file, vector<float> &acc, vector<float> &cond_acc) {
  vector<float>::iterator it;
  ifstream ifs;
  int s;

  ifs.open(input_file.c_str(), ios::in | ios::binary);
  if (!ifs) {
    cout << "Error: can't open acc_file: " << input_file << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ifs.read(reinterpret_cast<char*>(&s), sizeof(int));
  acc.assign(s, 0.0);
  for (it = acc.begin(); it != acc.end(); it++) {
      ifs.read(reinterpret_cast<char*>(&*it), sizeof(float));
  }
  ifs.read(reinterpret_cast<char*>(&s), sizeof(int));
  cond_acc.assign(s, 0.0);
  for (it = cond_acc.begin(); it != cond_acc.end(); it++) {
      ifs.read(reinterpret_cast<char*>(&*it), sizeof(float));
  }

  ifs.close();

  if (remove(input_file.c_str()) != 0) {
    cout << "Warning: there was an error deleting acc_file: " << input_file << "." << endl;
  }
}

void balance_workload(const RnaInteractionSearchParameters parameters,
                      vector<string> &sequences, vector<string> &names,
                      vector<int> &idx, vector<acc_node> &nodes) {
  unordered_map<int, int> order;
  vector<int> indices;
  int rank, procs, size;
  int *counts, *displ;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  displ = (int *) malloc(procs * sizeof(int));
  counts = (int *) malloc(procs * sizeof(int));
  if (rank == 0) {
    for (int i = 0; i < procs; i++) {
      counts[i] = 0;
      displ[i] = (i == 0 ? 0 : displ[i - 1] + counts[i - 1]);
      for (int j = i; j < nodes.size(); j += procs) {
        indices.push_back(nodes[j].idx);
        counts[i] += 1;
      }
    }
  }
  MPI_Bcast(counts, procs, MPI_INT, 0, MPI_COMM_WORLD);
  idx.resize(counts[rank]);
  MPI_Scatterv(indices.data(), counts, displ, MPI_INT, idx.data(), idx.size(),
               MPI_INT, 0, MPI_COMM_WORLD);

  free(counts);
  free(displ);

  for (int i = 0; i < idx.size(); i++) {
    order[idx[i]] = i;
  }
  sort(idx.begin(), idx.end());
  
  vector<string> tmp_seqs;
  vector<string> tmp_names;
  vector<int> tmp_idx(idx.size());
  for (int i = 0; i < idx.size(); i++) {
    tmp_idx[i] = idx[i];
  }

  FastafileReader fastafile_reader;
  fastafile_reader.read_file(parameters.GetInputFilename(), tmp_seqs, tmp_names,
                             tmp_idx);

  sequences.resize(idx.size());
  names.resize(idx.size());
  for (int i = 0; i < idx.size(); i++) {
    int pos = order[tmp_idx[i]];

    sequences[pos] = tmp_seqs[i];
    names[pos] = tmp_names[i];
    idx[pos] = tmp_idx[i];
  }
}

void RnaInteractionSearch::Run(const RnaInteractionSearchParameters parameters) {
  DbReader db_reader(parameters.GetDbFilename(), parameters.GetHashSize());
  double my_read, my_search, my_write[2];
  double read, search, write[2];
  int rank, procs, threads;
  int *last;
  vector<string> sequences;
  vector<string> names;
  vector<int> idx;

  threads = omp_get_max_threads();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  // create a shared variable
  if (rank == 0) {
    MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &last);

    if (last == NULL) {
      cout << "Fatal error: cannot allocate memory." << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *last = -1;
  } else {
    last = NULL;
  }

  MPI_Win win;
  MPI_Win_create(last, (rank == 0 ? sizeof(int) : 0), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  if (parameters.GetDebug()) my_read = MPI_Wtime();

  // create acc dir
  string shared_dir = acc_dir(parameters.GetSharedPath());
  if (rank == 0) {
    if (mkdir(shared_dir.c_str(), 0777) == -1) {
      cout << "Fatal error: cannot create acc directory." << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  ReadFastaFile(parameters, sequences, names, idx);

  // sort sequences according to their size
  vector<int> indices(sequences.size());
  for (int i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }
  sort(indices.begin(), indices.end(), sort_indices(&sequences));

  if (parameters.GetDebug()) {
    my_read = MPI_Wtime() - my_read;
    MPI_Reduce(&my_read, &read, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }

  if (parameters.GetDebug()) my_search = MPI_Wtime();
  if (rank == 0) cout << "pRIblast has started." << endl;

  vector<string> acc_files(threads);
  vector<string> output_files(threads);
  for (int i = 0; i < threads; i++) {
    string out_file = my_output_file(rank, i, parameters.GetPath());
    string acc_file = my_acc_file(shared_dir, rank, i);
    output_files[i] = out_file;
    acc_files[i] = acc_file;
    create_output_file(out_file);
    create_output_file(acc_file);
    
  }

  #pragma omp parallel
  {
    ofstream ofs;

    ofs.open(acc_files[omp_get_thread_num()], ios::out);
    if (!ofs) {
      cout << "Fatal error: can't open acc_file: " << acc_files[omp_get_thread_num()] << ".\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    #pragma omp for schedule(dynamic)
    for (int i = 0; i < sequences.size(); i++) {
      vector<float> query_cond_accessibility;
      vector<float> query_accessibility;

      int ii = indices[i];

      string query_sequence = sequences[ii];
      string path = acc_file(shared_dir, idx[ii]);

      CalculateAccessibility(parameters, query_sequence, query_accessibility,
	                     query_cond_accessibility);
      float f = save_acc(path, query_accessibility, query_cond_accessibility);
      
      ofs << idx[ii] << " " << f / query_sequence.size() << "\n";
    }

    ofs << "-1 0.0\n";
    ofs.close();
  }

  sequences.clear();
  indices.clear();
  names.clear();
  idx.clear();

  MPI_Barrier(MPI_COMM_WORLD);

  vector<acc_node> acc_nodes;
  if (rank == 0) {
    for (int i = 0; i < procs; i++) {
      for (int j = 0; j < threads; j++) {
        string input_file = my_acc_file(shared_dir, i, j);
        ifstream ifs;

        ifs.open(input_file, ios::in);
        if (!ifs) {
          cout << "Fatal error: cannot open acc_info_file: " << input_file << ".\n";
          MPI_Abort(MPI_COMM_WORLD, 1);
        }

        float ff;
        int ii;
        
        while (1) {
          ifs >> ii;
          ifs >> ff;
          
          if (ii == -1) {
            break;
          }
          
          acc_nodes.push_back(acc_node(ii, ff));
        }
        ifs.close();
        if (remove(input_file.c_str()) != 0) {
          cout << "Warning: could not delete acc_file: " << input_file << ".\n";
        }
      }
    }
    sort(acc_nodes.begin(), acc_nodes.end());
  }
  balance_workload(parameters, sequences, names, idx, acc_nodes);
  _dbs = db_reader.load_dbs();

  #pragma omp parallel for schedule(dynamic)
  for(int i = 0; i < sequences.size(); i++) {
    vector<float> query_conditional_accessibility;
    vector<unsigned char> query_encoded_sequence;
    vector<float> query_accessibility;
    vector<int> query_suffix_array;
    vector<Hit> hit_result;

    string query_sequence = sequences[i];
    string query_name = names[i];

    string path = acc_file(shared_dir, idx[i]);
    load_acc(path, query_accessibility, query_conditional_accessibility);
    ConstructSuffixArray(parameters, query_sequence, query_encoded_sequence, query_suffix_array);
    int length_count = 0;
    for (int j= 0; j < query_encoded_sequence.size(); j++) {
      if (query_encoded_sequence[j] >= 2 && query_encoded_sequence[j] <= 5) {
        length_count++;
      }
    }

    for (int j = 0; j < _dbs.size(); j++) {
      SearchSeed(parameters, hit_result, query_encoded_sequence, query_suffix_array, query_accessibility, query_conditional_accessibility, j);
      ExtendWithoutGap(parameters,hit_result, query_encoded_sequence, query_accessibility, query_conditional_accessibility, j);
      ExtendWithGap(parameters,hit_result, query_encoded_sequence, query_accessibility, query_conditional_accessibility, j);
      Output(parameters, hit_result, query_name, length_count, output_files[omp_get_thread_num()], j);

      hit_result.clear();
    }
  }

  if (parameters.GetDebug()) {
    my_search = MPI_Wtime() - my_search;
    my_write[0] = MPI_Wtime();
  }

  // write to the output file individually
  int tmp, src, data[2];
  if (procs > 1) {
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    MPI_Fetch_and_op(&rank, &tmp, MPI_INT, 0, 0, MPI_REPLACE, win);
    MPI_Win_unlock(0, win);
  } else {
    // using rdma as the underlying framework to implement rma operations may
    // result in runtime exceptions when calling fetch_and_op when there is
    // only one process in the communicator
    // newer versions of the framework may have fixed the issue, anyway ill
    // leave this sanity check here in case the program is run in old
    // environments
    // it is possible to use pt2pt instead of rdma and avoid this check
    // mpirun ... --mca osc pt2pt ...
    // however pt2pt's implementation does not perform as good
    tmp = -1;
  }

  if (tmp != -1) {
    MPI_Send(&rank, 1, MPI_INT, tmp, tmp, MPI_COMM_WORLD);
    MPI_Recv(&data[0], 2, MPI_INT, tmp, tmp, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    data[0] = 0;
    data[1] = 0;
  }

  data[0] = merge_output(parameters, rank, threads, data[0], data[1] == 0);

  if (++data[1] != procs) {
    MPI_Recv(&src, 1, MPI_INT, MPI_ANY_SOURCE, rank, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Send(&data[0], 2, MPI_INT, src, rank, MPI_COMM_WORLD);
  }

  if (parameters.GetDebug()) {
    my_write[1] = MPI_Wtime();

    MPI_Reduce(&my_search, &search, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_write[0], &write[0], 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      cout << "Time spent reading: " << read << " seconds." << endl;
      cout << "Time spent processing: " << search << " seconds." << endl;
      cout << "Time spent writing: " << write[1] - write[0] << " seconds." << endl;
      cout << "Total time: " << read + search + write[1] - write[0] << " seconds." << endl;
      cout.flush();
    }
  }

  MPI_Win_free(&win);
  if (rank == 0) {
    MPI_Free_mem(last);
    remove(shared_dir.c_str());
    cout << "pRiblast has ended." << endl;
  }
}

void RnaInteractionSearch::ReadFastaFile(const RnaInteractionSearchParameters parameters, vector<string> &sequences, vector<string> &names, vector<int> &idx){
  FastafileReader fastafile_reader;
  fastafile_reader.ReadFastafile(parameters.GetInputFilename(), sequences, names, idx);
};

void RnaInteractionSearch::CalculateAccessibility(const RnaInteractionSearchParameters parameters, string &query_sequence, vector<float> &query_accessibility, vector<float> &query_conditional_accessibility){
  FastafileReader fastafile_reader;
  Raccess raccess(parameters.GetMaximalSpan(), parameters.GetMinAccessibleLength());
  raccess.Run(query_sequence, query_accessibility, query_conditional_accessibility);
};

void RnaInteractionSearch::ConstructSuffixArray(const RnaInteractionSearchParameters parameters, string &query_sequence,  vector<unsigned char> &query_encoded_sequence, vector<int> &query_suffix_array){
  Encoder encoder(parameters.GetRepeatFlag());
  encoder.Encode(query_sequence, query_encoded_sequence, 0);
  query_suffix_array.resize(query_encoded_sequence.size());
  sais(&query_encoded_sequence[0], &query_suffix_array[0], query_encoded_sequence.size());
};

void RnaInteractionSearch::SearchSeed(const RnaInteractionSearchParameters parameters, vector<Hit> &hit_result, vector<unsigned char> &query_encoded_sequence, vector<int> &query_suffix_array, vector<float> &query_accessibility, vector<float> &query_conditional_accessibility, int idx){
  SeedSearch seed_search(parameters.GetHashSize(), parameters.GetMaxSeedLength(), parameters.GetMinAccessibleLength(), parameters.GetHybridEnergyThreshold());

  seed_search.Run(query_encoded_sequence, query_suffix_array, _dbs[idx].get_seqs(), _dbs[idx].get_suffix_array(), _dbs[idx].get_start_hash(), _dbs[idx].get_end_hash());
  seed_search.CalcInteractionEnergy(hit_result, query_suffix_array, _dbs[idx].get_suffix_array(), query_accessibility, query_conditional_accessibility, _dbs[idx].get_access(), _dbs[idx].get_cond_access(), _dbs[idx].get_seq_length(), _dbs[idx].get_start_pos());
}

void RnaInteractionSearch::ExtendWithoutGap(const RnaInteractionSearchParameters parameters, vector<Hit> &hit_result, vector<unsigned char> &query_encoded_sequence, vector<float> &query_accessibility, vector<float> &query_conditional_accessibility, int idx){
  UngappedExtension ungapped_extension(parameters.GetMinAccessibleLength(), parameters.GetDropOutLengthWoGap());

  ungapped_extension.Run(hit_result, query_encoded_sequence, _dbs[idx].get_seqs(), query_accessibility, query_conditional_accessibility, _dbs[idx].get_access(), _dbs[idx].get_cond_access());
  sort(hit_result.begin(), hit_result.end(), compare);
  CheckRedundancy(hit_result, parameters.GetInteractionEnergyThreshold());
  GetBasePair(hit_result, query_encoded_sequence, idx);
}

void RnaInteractionSearch::ExtendWithGap(const RnaInteractionSearchParameters parameters, vector<Hit> &hit_result, vector<unsigned char> &query_encoded_sequence, vector<float> &query_accessibility, vector<float> &query_conditional_accessibility, int idx){
  GappedExtension gapped_extension(parameters.GetMinAccessibleLength(), parameters.GetDropOutLengthWGap(), parameters.GetMinHelixLength());
  gapped_extension.Run(hit_result, query_encoded_sequence, _dbs[idx].get_seqs(), query_accessibility, query_conditional_accessibility, _dbs[idx].get_access(), _dbs[idx].get_cond_access(), _dbs[idx].get_seq_length());
  for (int i = 1; i < hit_result.size(); i++){
    hit_result[i].SortBasePair();
  }
  sort(hit_result.begin(), hit_result.end(), compare);
  CheckRedundancy(hit_result, parameters.GetFinalThreshold());
}

void RnaInteractionSearch::Output(const RnaInteractionSearchParameters parameters, vector<Hit> &hit_result, string q_name, int q_length, string output_file, int idx) {
  ofstream ofs;

  ofs.open(output_file.c_str(), ios::app);
  if (!ofs){
    cout << "Error: can't open output_file: " << output_file << "." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int output_style = parameters.GetOutputStyle();
  for(int i = 0; i < hit_result.size(); i++){
    int id = hit_result[i].GetDbSeqId();
    int db_length = _dbs[idx].get_seq_length_rep()[id];
    int seq_start_position = _dbs[idx].get_start_pos()[id];

    ofs << q_name << ",";
    ofs << q_length << ",";
    ofs << _dbs[idx].get_names()[id] << ",";
    ofs << db_length << ",";
    ofs << hit_result[i].GetAccessibilityEnergy() << ",";
    ofs << hit_result[i].GetHybridizationEnergy() << ",";
    ofs << hit_result[i].GetEnergy() << ",";

    db_length = _dbs[idx].get_seq_length()[id];
    int basepair_length = hit_result[i].GetBasePairLength();
    if (output_style == 1) {
      for(int j = 0; j < basepair_length; j++){
	    int dbpos = (db_length - 1) - (hit_result[i].GetBasePairSecond(j) - seq_start_position);
	    ofs << "(" << hit_result[i].GetBasePairFirst(j) << ":" << dbpos << ") ";
      }
    } else {
      ofs << "(" << hit_result[i].GetBasePairFirst(0) << "-" << hit_result[i].GetBasePairFirst(basepair_length - 1) << ":";
      int dbpos1 = (db_length - 1) - (hit_result[i].GetBasePairSecond(0) - seq_start_position);
      int dbpos2 = (db_length - 1) - (hit_result[i].GetBasePairSecond(basepair_length - 1) - seq_start_position);
      ofs << dbpos1 << "-" << dbpos2 << ") ";
    }
    ofs << endl;
  }
  ofs.close();
}

void RnaInteractionSearch::GetBasePair(vector<Hit> &hit_result, vector<unsigned char> &query_encoded_sequence, int idx) {
  for(int i = 0; i< hit_result.size(); i++){
    int q_start = hit_result[i].GetQSp();
    int db_start = hit_result[i].GetDbSp();
    int length = hit_result[i].GetQLength();
    for(int j = 0; j<length;j++){
      if(BP_pair[query_encoded_sequence[q_start+j]-1][_dbs[idx].get_seqs()[db_start+j]-1] != 0){
	hit_result[i].AddBasePair(q_start+j, db_start+j);
      }
    }
  }
}

void RnaInteractionSearch::CheckRedundancy(vector<Hit> &hit_result, double energy_threshold){
  for(int i = 0; i<hit_result.size();i++){
    if(hit_result[i].GetEnergy() > energy_threshold){
      hit_result[i].SetFlag();
    }

    if(!hit_result[i].GetFlag()){
      int a_QSp = hit_result[i].GetQSp();
      int a_DbSp = hit_result[i].GetDbSp();
      int a_QEp = a_QSp+hit_result[i].GetQLength()-1;
      int a_DbEp = a_DbSp+hit_result[i].GetDbLength()-1;

      for(int j = i+1; j<hit_result.size();j++){
        if(!hit_result[j].GetFlag()){
          int b_DbSp = hit_result[j].GetDbSp();
          if(a_DbEp < b_DbSp){
            break;
          }

          int b_QSp = hit_result[j].GetQSp();
          int b_QEp = b_QSp+hit_result[j].GetQLength()-1;
          int b_DbEp = b_DbSp+hit_result[j].GetDbLength()-1;
          if(a_QEp>=b_QEp && a_QSp <= b_QSp && a_DbEp >= b_DbEp){
	    if(hit_result[i].GetEnergy() > hit_result[j].GetEnergy()){
	      hit_result[i].SetFlag();
	    }else{
	      hit_result[j].SetFlag();
	    }
          }
        }
      }
    }
  }
  hit_result.erase(remove_if(hit_result.begin(), hit_result.end(), CheckFlag()), hit_result.end());

}
