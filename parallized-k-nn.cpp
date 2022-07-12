#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <sys/resource.h>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <future>
#include <functional>
#include <queue>
#include <utility>
#include <fstream>
#include <chrono>

// Global variables
class KD_Node;
static unsigned seed = 129;
static uint64_t n_dim;
KD_Node * nodes;               // preallocate the nodes
std::atomic<unsigned> n_index; // to keep track which 2 nodes to assign to the next thread

/* -------------------- Reader -------------------- */
// Wrapper around a pointer, for reading values from byte sequence.
class Reader {
    public:
        Reader(const char *p) : ptr{p} {}
        template <typename T>
        Reader &operator>>(T &o) {
            // Assert alignment.
            assert(uintptr_t(ptr)%sizeof(T) == 0);
            o = *(T *) ptr;
            ptr += sizeof(T);
            return *this;
        }
    private:
        const char *ptr;
};

/* -------------------- Thread Safe Queue -------------------- */
template <typename T>
class my_Queue {
    public:
        my_Queue () {};
        bool empty() {
            std::unique_lock<std::mutex> lock(q_mutex);
            return safe_queue.empty();
        }
        int size() {
            std::unique_lock<std::mutex> lock(q_mutex);
            return safe_queue.size();
        }
        void push(T job) {
            std::unique_lock<std::mutex> lock(q_mutex);
            safe_queue.push(job);
        }

        // returns a bool if there is a job in the queue
        // if there is a job assign to the argument passed in
        bool pop(T& job) {
            std::unique_lock<std::mutex> lock(q_mutex);
            if (safe_queue.empty()) {return false;}
            job = std::move(safe_queue.front());
            safe_queue.pop();
            return true;
        }
    private:
        std::queue<T> safe_queue;
        std::mutex q_mutex;
};


/* -------------------- Thread Pool -------------------- */
/*
  Constructing KD tree using ThreadPool:
    - Each thread will construct a node and continue constructing down the left child while enqueuing another job in the thread pool to construct the right child
  Querying KNN:
    - Worker threads will search through the constructed tree and store the results within this thread pool, which will be periodically iterated through by a main thread and written to a file
*/
class Thread_Pool {
  public:
    // Constructor and destructor
    Thread_Pool(unsigned n_threads, unsigned size) : done(false), num_task{0}, num_working_threads{0} {
        KNN_results.resize(size);
        create_threads_construct_kd(n_threads);
    }
    ~Thread_Pool(); // join all threads and change the done to true.

    // storing the finished knn queries
    my_Queue<unsigned> KNN_finished_indices;
    std::vector<std::vector<std::vector<float>>> KNN_results;
    void enque_construction_jobs(std::function<void ()>);
  private:

    friend class KD_Node;
    std::mutex queue_mutex;
    std::atomic<bool> done;
    std::atomic<int> num_task;
    std::atomic<int> num_working_threads;
    std::vector<std::thread> work_threads;
    std::condition_variable condition_var;
    my_Queue<std::function<void()>> tasks_q;

    // create threads and have them wait until there are jobs in the queue
    void create_threads_construct_kd(int);
    void multithread_construct_KD_tree_wait_q();
};

/* Create the threads to be used in the thread pool */
void Thread_Pool::create_threads_construct_kd(int n_threads) {
  for (int i = 0; i < n_threads; i++) {
    work_threads.push_back(std::thread(&Thread_Pool::multithread_construct_KD_tree_wait_q, this));
  }
}

/* Enqueue jobs for constructing the K-D tree */
void Thread_Pool::enque_construction_jobs(std::function<void ()> job) {
  num_task++;
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    tasks_q.push(job);
  }
  condition_var.notify_one();
}

/* All the threads yield until there is an available job  */
void Thread_Pool::multithread_construct_KD_tree_wait_q() {
  while (1) {
    std::function<void()> task;
    bool valid_job;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      condition_var.wait(lock, [=]{return !tasks_q.empty() || done;});
      if (done && tasks_q.empty())
        return;
      valid_job = tasks_q.pop(task);
    }
    if (valid_job) {
      num_working_threads++;
      task();
      num_task--;
      num_working_threads--;
    } else {
      std::this_thread::yield();
    }
  }
}

/* Destructor for Thread_Pool, notify all threads and join them together */
Thread_Pool::~Thread_Pool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    done = true;
  }
  // Notifies all of the threads and joins them together
  condition_var.notify_all();
  for(auto &worker: work_threads) {
    worker.join();
  }
}


/* -------------------- KD Node -------------------- */
class KD_Node {
    public:
        KD_Node() : left_child{nullptr}, right_child{nullptr} {};
        KD_Node(float axis_val) : split_axis_val{axis_val}, left_child{nullptr}, right_child{nullptr} {};
        KD_Node(std::vector<std::vector<float>> data) : leaves{data}, left_child{nullptr}, right_child{nullptr} {};
        static void build_KD_Tree_multi_thread(float[], unsigned, KD_Node *, Thread_Pool *);
        static std::vector<std::vector<std::vector<float>>> KKN_multiple_thread(KD_Node *, float *, unsigned, unsigned, Thread_Pool *, char*);

        // For development purposes
        static int verify(KD_Node *, unsigned);
        static std::vector<std::vector<std::vector<float>>> KNN_check(KD_Node *, float *, unsigned, unsigned);
        static KD_Node * build_KD_Tree_single_thread(float[], unsigned);
        static std::vector<std::vector<std::vector<float>>> KNN_single_thread(KD_Node *, float *, unsigned, unsigned, unsigned);
    private:
        friend class Thread_Pool;
        float split_axis_val;
        std::vector<std::vector<float>> leaves;
        KD_Node *left_child, *right_child; // left is <= to the value and right is >
        static KD_Node * helper_KD_Tree(std::vector<float>, unsigned, unsigned);
        static void helper_KD_Tree_multi_thread(std::vector<float>, unsigned, unsigned, KD_Node *, Thread_Pool *);
        static void KNN_single_thread_helper(KD_Node *, std::vector<float>, unsigned, unsigned, std::priority_queue<std::pair<float, std::vector<float>>> &);
        static void KNN_check_helper(KD_Node *, std::vector<float>, unsigned, unsigned, std::priority_queue<std::pair<float, std::vector<float>>> &);
        static void KKN_multiple_main_thread(KD_Node *, std::vector<float>, unsigned, unsigned, Thread_Pool *);
        static void KKN_multiple_helper_thread(KD_Node *, std::vector<float>, unsigned, unsigned, std::priority_queue<std::pair<float, std::vector<float>>> &);
};

// build KD tree using thread pool, calls helper_KD_Tree_multi_thread
void KD_Node::build_KD_Tree_multi_thread(float * training, unsigned points, KD_Node * root, Thread_Pool * thread_pool) {
  std::vector<float> training_data(training, training + (points * n_dim));
  KD_Node::helper_KD_Tree_multi_thread(training_data, 0, points, root, thread_pool);
  //Need some synchronization mech for thread pool to notify the main thread
  while(thread_pool->num_task != 0 || !thread_pool->tasks_q.empty() || thread_pool->num_working_threads != 0) {
    std::this_thread::yield();
  }
}

// We will be passing in a node that is already created but need to fill out the information of it
void KD_Node::helper_KD_Tree_multi_thread(std::vector<float> data, unsigned axis, unsigned points, KD_Node * node, Thread_Pool * thread_pool) {
    // base case
  if (points <= 10) {
    std::vector<std::vector<float>> data_2d;
    for (unsigned i = 0; i < points; i++) {
      std::vector<float> temp = {};
      for (unsigned j = 0; j < n_dim; j++) {
        temp.push_back(data[(i*n_dim)+j]);
      }
      data_2d.push_back(temp);
    }
    node->leaves = data_2d;
    return;
  }

  float median;
  bool flag = true;

  // sampling and finding the median
  if (points <= 1000) {
    // sort it and pass the array into the next function?
    std::vector<float> sample;
    for (unsigned i = 0; i < points; i++) {
        sample.push_back(data[(i*n_dim) + axis]);
    }

    // finding the median of this axis
    std::nth_element(sample.begin(), sample.begin() + sample.size()/2, sample.end());
    median = sample[sample.size()/2];

    flag = false;

  } else {
    // extract a sample
    std::vector<float> sample;
    // Not sure if this will generate a random number every time...
    // Solution read in Dev/Random everytime --> assign a random seed to each thread or thread_local?
    thread_local unsigned thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    //unsigned thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    for (int i = 0; i < 1000; ++i) {
        sample.push_back(data[((rand_r(&thread_id) % points) * n_dim) + axis]);
    }
    // finding the median
    std::nth_element(sample.begin(), sample.begin() + sample.size()/2, sample.end());
    median = sample[sample.size()/2];
  }

  std::vector<float> left_data;
  std::vector<float> right_data;

  for (unsigned i = 0; i < points; i++) {
    if (data[(i*n_dim) + axis] <= median) {
      for (unsigned j = 0; j < n_dim; j++) {
        left_data.push_back(data[(i*n_dim) + j]);
      }
    } else {
      for (unsigned j = 0; j < n_dim; j++) {
        right_data.push_back(data[(i*n_dim) + j]);
      }
    }
  }

  node->split_axis_val = median;
  unsigned s_child_index = n_index.fetch_add(2);
  node->left_child = new (nodes + s_child_index) KD_Node{};
  node->right_child = new (nodes + s_child_index + 1) KD_Node{};

  // if we want to split this more put one half of it on the thread pool
  if (flag) {
    thread_pool->enque_construction_jobs(std::bind(helper_KD_Tree_multi_thread, left_data, (axis+1)%n_dim, left_data.size() / n_dim, node->left_child, thread_pool));
  } else {
    KD_Node::helper_KD_Tree_multi_thread(left_data, (axis+1)%n_dim, left_data.size() / n_dim, node->left_child, thread_pool);
  }

  KD_Node::helper_KD_Tree_multi_thread(right_data, (axis+1)%n_dim, right_data.size() / n_dim, node->right_child, thread_pool);

  return;
}

// computes the distance between two points
float distance (std::vector<float> point1, std::vector<float> point2) {
    float dis = 0.0;
    for (unsigned i = 0; i < point1.size(); i++) {
        dis += pow((point1[i] - point2[i]), 2);
    }
    return dis;
}

/*
  !  This assumes the ndim is the same for the knn query and building tree
  Idea behind multithreaded KNN search:
    - The main thread creates these calls;
    - A body function that the threads will call (passes in a priority_queue which is made within the local_thread memory) --> calls a helper function
        - Helper function: to do all the traversing
        - Updates the priority_queue
    - The body function has an updated priority_queue, which is able to index into the vector and place the knn
*/
std::vector<std::vector<std::vector<float>>> KD_Node::KKN_multiple_thread(KD_Node * root, float * query_array, unsigned num_queries, unsigned num_neighbors, Thread_Pool * thread_pool, char* result_f_name) {
    for (unsigned i = 0 ; i < num_queries; i++) {
        std::vector<float> query_point(query_array + (i*n_dim), query_array + (i*n_dim) + n_dim);
        thread_pool->enque_construction_jobs(std::bind(KKN_multiple_main_thread, root, query_point, num_neighbors, i, thread_pool));
    }

    int result_file = open(result_f_name, O_WRONLY | O_CREAT);
    unsigned offset = 56;
    unsigned num_queries_wrote = 0;

    // main thread checks if there any finished queries, if so it will write it to the result file
    while(num_queries_wrote < num_queries) {
        if (thread_pool->KNN_finished_indices.empty()) {
            std::this_thread::yield();
        } else {
            unsigned knn_finished_index;
            thread_pool->KNN_finished_indices.pop(knn_finished_index);
            std::vector<std::vector<float>> KNN_to_insert = thread_pool->KNN_results[knn_finished_index];
            
            // iterate through the num_neighbors
            for (unsigned i = 0; i < KNN_to_insert.size(); i++) {
                // iterate through the individual points
                for (unsigned j = 0; j < KNN_to_insert[i].size(); j++) {
                    pwrite(result_file, &KNN_to_insert[i][j], sizeof(float), offset + (num_neighbors * n_dim * knn_finished_index * 4) + (n_dim * i * 4) + (j * 4));
                }
            }
            num_queries_wrote++;
        }
    }
    close(result_file);
    return thread_pool->KNN_results;
}

// Calls the helper and places the result back into the result vector
void  KD_Node::KKN_multiple_main_thread(KD_Node * root, std::vector<float> query_point, unsigned num_neighbors, unsigned index, Thread_Pool * thread_pool) {
    thread_local std::priority_queue<std::pair<float, std::vector<float>>> pq_knn;
    pq_knn = std::priority_queue<std::pair<float, std::vector<float>>>();
    KD_Node::KKN_multiple_helper_thread(root, query_point, num_neighbors, 0, pq_knn);
    std::vector<std::vector<float>> temp_knn;
    while(!pq_knn.empty()){
        temp_knn.push_back(pq_knn.top().second);
        pq_knn.pop();
    }
    thread_pool->KNN_results[index] = temp_knn;
    thread_pool->KNN_finished_indices.push(index);
}


// Does the actual KNN search
void KD_Node::KKN_multiple_helper_thread(KD_Node * node, std::vector<float> query_point, unsigned num_neighbors, unsigned axis, std::priority_queue<std::pair<float, std::vector<float>>> & pq_knn) {
    // base case we hit a leaf
    if (node->left_child == nullptr && node->right_child == nullptr) {
        for (unsigned i = 0; i < node->leaves.size(); i++) {
            if ((num_neighbors > pq_knn.size())) {
                pq_knn.push(std::make_pair(distance(query_point, node->leaves[i]), node->leaves[i] ));
            } else if (num_neighbors <= pq_knn.size() && pq_knn.top().first > distance(query_point, node->leaves[i])) {
                pq_knn.pop();
                pq_knn.push(std::make_pair(distance(query_point, node->leaves[i]), node->leaves[i] ));
            }
            // }
        }
        return ;
    }

    // first recurision go down the tree until hit a leaf
    bool left = true;
    if (query_point[axis] <= node->split_axis_val) {
        KD_Node::KKN_multiple_helper_thread(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);
    } else {
        left = false;
        KD_Node::KKN_multiple_helper_thread(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);
    }

    // check if we need more neighbors
    if (pq_knn.size() < num_neighbors) {
        if (left) {KD_Node::KKN_multiple_helper_thread(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);}
        else {KD_Node::KKN_multiple_helper_thread(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);}
    } else {
        // check the top distance against this node's axis
        std::vector<float> line (query_point.begin(), query_point.end());
        line[axis] = node->split_axis_val;
        if (distance(query_point, line) < pq_knn.top().first) {
            if (left) { KD_Node::KKN_multiple_helper_thread(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn); }
            else { KD_Node::KKN_multiple_helper_thread(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn); }
        }
    }

}


/* For development purposes  */
/* 
  Single thread constructing KD tree
  ! No longer used. This was used to give inspiration for multithread build of kd tree 
 */
KD_Node* KD_Node::build_KD_Tree_single_thread(float * training,  unsigned points) {
  std::vector<float> training_data(training, training + (points * n_dim));
  return helper_KD_Tree(training_data, 0, points);
}

/* 
  Single thread constructing KD tree helper function
  ! No longer used. This was used to give inspiration for multithread build of kd tree 
 */
KD_Node * KD_Node::helper_KD_Tree(std::vector<float> data, unsigned axis, unsigned points) {

  // base case
  if (points <= 10) {
    std::vector<std::vector<float>> data_2d;
    for (unsigned i = 0; i < points; i++) {
      std::vector<float> temp;
      for (unsigned j = 0; j < n_dim; j++) {
        temp.push_back(data[(i*n_dim)+j]);
      }
      data_2d.push_back(temp);
    }
    return new KD_Node(data_2d);
  }

  float median;
  // sampling and finding the median
  if (points <= 1000) {
    // sort it and pass the array into the next function?
    std::vector<float> sample;
    for (unsigned i = 0; i < points; i++) {
        sample.push_back(data[(i*n_dim) + axis]);
    }

    // finding the median of this axis
    std::nth_element(sample.begin(), sample.begin() + sample.size()/2, sample.end());
    median = sample[sample.size()/2];

  } else {
    // extract a sample
    std::vector<float> sample;
    for (int i = 0; i < 1000; ++i) {
        unsigned random_index = ((rand_r(&seed) % points) * n_dim) + axis;
        sample.push_back(data[random_index]);
    }
    // finding the median
    std::nth_element(sample.begin(), sample.begin() + sample.size()/2, sample.end());
    median = sample[sample.size()/2];
  }

  std::vector<float> left_data;
  std::vector<float> right_data;

  for (unsigned i = 0; i < points; i++) {
    if (data[(i*n_dim) + axis] <= median) {
      for (unsigned j = 0; j < n_dim; j++) {
        left_data.push_back(data[(i*n_dim) + j]);
      }
    } else {
      for (unsigned j = 0; j < n_dim; j++) {
        right_data.push_back(data[(i*n_dim) + j]);
      }
    }
  }

  KD_Node * new_data = new KD_Node(median);
  new_data->left_child = helper_KD_Tree(left_data, (axis+1)%n_dim, left_data.size() / n_dim);
  new_data->right_child = helper_KD_Tree(right_data, (axis+1)%n_dim, right_data.size() / n_dim);
  return new_data;

}

/* 
  Single thread KNN
  ! No longer used. This was used to give inspiration for multithread knn 
 */
std::vector<std::vector<std::vector<float>>> KD_Node::KNN_single_thread(KD_Node * root, float * query_list, unsigned num_queries, unsigned num_dim, unsigned num_neighbors) {
    std::vector<std::vector<std::vector<float>>> vector_KNN_queries;
    for (unsigned i = 0; i < num_queries * num_dim; i += num_dim) {
         std::vector<float> query_point(query_list + i, query_list + i + num_dim);
         std::priority_queue<std::pair<float, std::vector<float>>> pq_knn;
        KD_Node::KNN_single_thread_helper(root, query_point, num_neighbors, 0, pq_knn);
        std::vector<std::vector<float>> temp_knn;
        while(!pq_knn.empty()){
            temp_knn.push_back(pq_knn.top().second);
            pq_knn.pop();
        }
        vector_KNN_queries.push_back(temp_knn);
    }
    return vector_KNN_queries;
}

/* 
  Single thread KNN helper function
  ! No longer used. This was used to give inspiration for multithread knn 
 */
void KD_Node::KNN_single_thread_helper(KD_Node * node, std::vector<float> query_point, unsigned num_neighbors, unsigned axis, std::priority_queue<std::pair<float, std::vector<float>>> &pq_knn) {
    // base case we hit a leaf
    if (node->left_child == nullptr && node->right_child == nullptr) {
        for (unsigned i = 0; i < node->leaves.size(); i++) {
            if (num_neighbors > pq_knn.size()) {
                pq_knn.push(std::make_pair(distance(query_point, node->leaves[i]), node->leaves[i] ));
            } else if (num_neighbors <= pq_knn.size() && pq_knn.top().first > distance(query_point, node->leaves[i])) {
                pq_knn.pop();
                pq_knn.push(std::make_pair(distance(query_point, node->leaves[i]), node->leaves[i] ));
            }
        }
        return ;
    }

    // first recurision go down the tree until hit a leaf
    bool left = true;
    if (query_point[axis] <= node->split_axis_val) {
        KD_Node::KNN_single_thread_helper(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);
    } else {
        left = false;
        KD_Node::KNN_single_thread_helper(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);
    }

    // check if we need more neighbors
    if (pq_knn.size() < num_neighbors) {
        if (left) {KD_Node::KNN_single_thread_helper(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);}
        else {KD_Node::KNN_single_thread_helper(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);}
    } else {
        // check the top distance against this node's axis
        std::vector<float> line (query_point.begin(), query_point.end());
        line[axis] = node->split_axis_val;
        if (distance(query_point, line) < pq_knn.top().first) {
            if (left) { KD_Node::KNN_single_thread_helper(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn); }
            else { KD_Node::KNN_single_thread_helper(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn); }
        }
    }

}

// checks all the leaves and tries to find the closest neighbor
void KD_Node::KNN_check_helper(KD_Node * node, std::vector<float> query_point, unsigned num_neighbors, unsigned axis, std::priority_queue<std::pair<float, std::vector<float>>> & pq_knn) {
        // base case we hit a leaf
    if (node->left_child == nullptr && node->right_child == nullptr) {
        for (unsigned i = 0; i < node->leaves.size(); i++) {
            if ((num_neighbors > pq_knn.size())) {
                pq_knn.push(std::make_pair(distance(query_point, node->leaves[i]), node->leaves[i] ));
            } else if (num_neighbors <= pq_knn.size() && pq_knn.top().first > distance(query_point, node->leaves[i])) {
                pq_knn.pop();
                pq_knn.push(std::make_pair(distance(query_point, node->leaves[i]), node->leaves[i]));
            }
            // }
        }
        return ;
    }

    if (node->left_child) {
        KD_Node::KNN_check_helper(node->left_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);
    }

    if (node->right_child) {
        KD_Node::KNN_check_helper(node->right_child, query_point, num_neighbors, (axis + 1) % n_dim, pq_knn);
    }
}

// checks the first few KNN Queries
std::vector<std::vector<std::vector<float>>> KD_Node::KNN_check(KD_Node * root, float * query_list, unsigned num_queries, unsigned num_neighbors) {
    std::vector<std::vector<std::vector<float>>> vector_KNN_queries;
    for (unsigned i = 0; i < num_queries * n_dim; i += n_dim) {
        //  printf("this\n");
         std::vector<float> query_point(query_list + i, query_list + i + n_dim);
         std::priority_queue<std::pair<float, std::vector<float>>> pq_knn;
        KD_Node::KNN_check_helper(root, query_point, num_neighbors, 0, pq_knn);
        std::vector<std::vector<float>> temp_knn;
        while(!pq_knn.empty()){
            // printf("hit\n");
            temp_knn.push_back(pq_knn.top().second);
            pq_knn.pop();
        }
        vector_KNN_queries.push_back(temp_knn);
    }
    return vector_KNN_queries;
}

// prints the KD_Tree out
int KD_Node::verify(KD_Node * node, unsigned depth) {
  if (!node->leaves.empty()) {
    int count_leaves = 0;
    //printf("This level %u is a leaf. It contains %lu points.\n", depth, node->leaves.size());
    for (unsigned i = 0; i < node->leaves.size(); i++) {
      //printf("Point %u:", i);
      for (unsigned j = 0; j < node->leaves[i].size(); j++) {
        //printf(" %f ", node->leaves[i][j]);
        count_leaves++;
      }
      //printf("\n");
    }
    return count_leaves;
  }
  //printf("This level %u divides on axis val: %f\n", depth, node->split_axis_val);
  int acc_leaves = 0;
  if (node->left_child != nullptr) {
    acc_leaves += verify(node->left_child, depth + 1);
  }
  if (node->right_child != nullptr) {
    acc_leaves += verify(node->right_child, depth + 1);
  }
  return acc_leaves;
}


/* -------------------- Helper functions -------------------- */
// Function that outputs the MMap file
void output_mmap_file(uint64_t training_file_id, uint64_t training_n_points, uint64_t n_dims, float * training_array) {
    // Prefix to print before every line, to improve readability.
    std::string pref("    ");

    std::cout << pref << "Training file ID: " << std::hex << std::setw(16) << std::setfill('0') << training_file_id << std::dec << std::endl;
    std::cout << pref << "Number of points: " << training_n_points << std::endl;
    std::cout << pref << "Number of dimensions: " << n_dims << std::endl;
    int point_counter = 0;
    for (std::uint64_t i = 0; i < training_n_points * n_dims; i++) {
        if (i % n_dims == 0) {
            std::cout << pref << "Point " << point_counter << ": ";
        }
        float f = training_array[i];
        std::cout << std::fixed << std::setprecision(6) << std::setw(15) << std::setfill(' ') << f;
        // Add comma.

        if (i % n_dims == 2) {
            point_counter += 1;
            std::cout << std::endl;
        } else {
            std::cout << ", ";
        }
    }
    std::cout << pref << "Point_counter: " << point_counter << std::endl;

}


void output_KNN_results(std::vector<std::vector<std::vector<float>>> vec_all_knn, float * query_array, unsigned num_print) {
  for (unsigned i = 0; i < num_print; i++) {
    printf("Query #%d of query point [", i);

    for (unsigned i_j = 0; i_j < n_dim; i_j++) {
      if (i_j == n_dim-1) {
        printf("%f", query_array[(i * n_dim) + i_j]);
      } else {
        printf("%f, ", query_array[(i * n_dim) + i_j]);
      }
    }
    printf("]:\n");

    for (unsigned j = 0; j < vec_all_knn[i].size(); j++) {
      printf("\tPoint #%d: [", j);
      for (unsigned k = 0; k < vec_all_knn[i][j].size(); k++) {
        if (k == vec_all_knn[i][j].size()-1) {
          printf("%f", vec_all_knn[i][j][k]);
        } else {
          printf("%f, ", vec_all_knn[i][j][k]);
        }
      }
      printf("] \n");
    }
  }
}


/* -------------------- Main -------------------- */
// ./k-nn n_cores training_file query_file result_file
int main(int argc, char * argv[]) {

    // Check the number of args
    if (argc != 5) {
        std::cerr << "Invalid Number of arguments. argc : ./k-nn n_cores training_file query_file result_file" << std::endl;
        return -1;
    }

    int rv;

    /*                Set Up (MMap Training File)               */
    // Extract the only needed information
    uint64_t training_file_id;
    uint64_t training_n_points;
    uint64_t n_dims;
    float *training_array = nullptr;
    {
      // Open file
      int fd = open(argv[2], O_RDONLY);
      if (fd < 0) {
          int en = errno;
          std::fprintf(stderr, "Couldn't open %s: %s\n", argv[1], strerror(en));
          exit(2);
      }

      // Get size of file
      struct stat sb;
      rv = fstat(fd, &sb); assert(rv == 0);

      // MMap file
      void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
      if (vp == MAP_FAILED) {
          int en = errno;
          fprintf(stderr, "mmap() failed: %s\n", strerror(en));
          exit(3);
      }

      rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);

      rv = close(fd); assert(rv == 0);

      char * file_mem = (char *) vp;
      Reader reader{file_mem + 8}; // From dump.cpp
      reader >> training_file_id >> training_n_points >> n_dims;

      training_array = (float *) (file_mem + 32);
    }

    /*                 Set Up (MMap Query File)                 */

    uint64_t query_file_id;
    uint64_t n_queries;
    uint64_t n_query_dims;
    uint64_t n_neighbors;
    float *query_array = nullptr;
    {
      // Open file
      int fd = open(argv[3], O_RDONLY);
      if (fd < 0) {
        int en = errno;
        std::fprintf(stderr, "Couldn't open %s: %s\n", argv[3], strerror(en));
        exit(2);
      }

      // Get size of file
      struct stat sb;
      rv = fstat(fd, &sb); assert(rv == 0);

      // MMap file
      void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
      if (vp == MAP_FAILED) {
        int en = errno;
        fprintf(stderr, "mmap() failed: %s\n", strerror(en));
        exit(3);
      }

      rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);

      rv = close(fd); assert(rv == 0);

      // Extract the necessary information
      char * file_mem = (char *) vp;
      Reader reader{file_mem + 8}; // From dump.cpp
      reader >> query_file_id >> n_queries >> n_query_dims >> n_neighbors;

      query_array = (float *) (file_mem + 40);
    }
    n_dim = n_dims;

    /*                    Constructing KD tree                  */

    // Get the CPU affinity
    cpu_set_t mask;
    unsigned nproc;
    sched_getaffinity(0, sizeof(cpu_set_t), &mask);
    nproc = sysconf(_SC_NPROCESSORS_ONLN); // maybe i should just follow this?
    unsigned num_threads = (nproc * 2); // I believe the servers have 2 threads per core
    //printf("\n-------------------------------------------------\n");

    // Using multiple threads to construct the tree
    Thread_Pool pool(num_threads, n_queries);
    nodes = (KD_Node *) ::operator new(training_n_points*sizeof(KD_Node));
    KD_Node * root_mult = new(nodes) KD_Node{};
    n_index = 1;

    // Timing the KD construction
    auto start = std::chrono::high_resolution_clock::now();
    KD_Node::build_KD_Tree_multi_thread(training_array, training_n_points, root_mult, &pool);
    std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
    std::cerr << "Accoumulated time for KD Tree Construction: " << elapsed.count() << std::endl;

    // Extra check that all the points are added to the tree
    // int mult_num_points = KD_Node::verify(root_mult, 0);
    // printf("After verifying, there are %d data points.\n", mult_num_points);
    // printf("The file specified, there are %lu data points (points * n_dim).\n", training_n_points * n_dims);
    // printf("\n-------------------------------------------------\n");


    /*             Creating New File/Writing Results                 */

    // Reading /dev/urandom to generate a random number
    unsigned long int result_file_id = 0;
    {
        size_t size = sizeof(result_file_id);
        std::ifstream urandom_f("/dev/urandom", std::ios::in|std::ios::binary);
        if (urandom_f) {
            urandom_f.read(reinterpret_cast<char*>(&result_file_id), size);
            urandom_f.close();
        } else {
            printf("Failed to open /dev/urandom");
        }
    }

    // Writing the heading information
    auto myfile = std::fstream(argv[4], std::ios::out | std::ios::binary | std::ios::trunc);
    char result[8] = {'R', 'E', 'S', 'U', 'L', 'T', '\0', 0};
    myfile.write(&result[0], sizeof(result));
    myfile.write((char *)&training_file_id, sizeof(training_file_id));
    myfile.write((char *)&query_file_id, sizeof(query_file_id));
    myfile.write((char *)&result_file_id, sizeof(result_file_id));
    myfile.write((char *)&n_queries, sizeof(n_queries));
    myfile.write((char *)&n_query_dims, sizeof(n_query_dims));
    myfile.write((char *)&n_neighbors, sizeof(n_neighbors));
    myfile.close();

    /*                    Performing KNN                        */

    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<float>>> mult_vec_all_knn = KD_Node::KKN_multiple_thread(root_mult, query_array, n_queries, n_neighbors, &pool, argv[4]);
    elapsed = std::chrono::system_clock::now() - start;
    std::cerr << "Accoumulated time for KNN: " << elapsed.count() << std::endl;

    /*                          Clean Up                             */
    for (unsigned i = 0; i < n_index; i++) {
        nodes[i].~KD_Node();
    }

    ::operator delete(nodes);

    /*                    Performance Results                        */
    struct rusage ru;
    rv = getrusage(RUSAGE_SELF, &ru); assert(rv == 0);
    auto cv = [](const timeval &tv) {
        return double(tv.tv_sec) + double(tv.tv_usec)/1000000;
    };
    std::cerr << "Resource Usage:\n";
    std::cerr << "    User CPU Time: " << cv(ru.ru_utime) << '\n';
    std::cerr << "    Sys CPU Time: " << cv(ru.ru_stime) << '\n';
    std::cerr << "    Max Resident: " << ru.ru_maxrss << '\n';
    std::cerr << "    Page Faults: " << ru.ru_majflt << '\n';

}