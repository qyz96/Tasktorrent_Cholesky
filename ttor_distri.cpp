#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>
#include <cmath>


#include <mpi.h>

using namespace std;
using namespace Eigen;
using namespace ttor;

using namespace std;
using namespace ttor;


typedef array<int, 2> int2;
typedef array<int, 3> int3;





//Test Test2
void cholesky(int n_threads, int verb, int n, int nb, int n_col, int n_row, int priority)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    std::atomic<long long int> potrf_us_t(0);
    std::atomic<long long int> trsm_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> accu_us_t(0);

    int q = static_cast<int>(cbrt(n_ranks));
    if (q * q * q != n_ranks)
    {
        if (rank == 0)
        {
            cerr << "Number of processes must be a perfect cube." << endl;
        }
        MPI_Finalize();
        exit(1);
    }

    int3 rank_3d;
    rank_3d[0] = rank / (q*q);
    rank_3d[1] = (rank % (q*q)) / q;
    rank_3d[2] = (rank % (q*q)) % q;


    // Number of tasks
    int n_tasks_per_rank = 2;

    
    struct acc_data {
        std::map<int, std::unique_ptr<MatrixXd>> to_accumulate; // to_accumulate[k] holds matrix result of gemm(k,i,j)
        std::mutex mtx; // Protects that map
    };





    std::vector<acc_data> gemm_results(nb*nb);


    auto rank3d21 = [&](int i, int j, int k) { return ((j % q) * q + k % q) + (i % q) * q * q;};
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd A;
    A = MatrixXd::NullaryExpr(n*nb,n*nb, val);
    MatrixXd L = A;
    vector<unique_ptr<MatrixXd>> blocs(nb*nb);

    auto bloc_2_rank = [&](int i, int j) {
        int r = (j % n_col) * n_row + (i % n_row);
        assert(r >= 0 && r < n_ranks);
        return r;
    };

    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            blocs[ii+jj*nb]=make_unique<MatrixXd>(n,n);
            if (rank == rank3d21(ii,jj, jj))   {
                *blocs[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
            }
            else {
                *blocs[ii+jj*nb] = MatrixXd::Zero(n,n);
            }
        }
    }



    // Map tasks to rank
    

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "WkTuto_" + to_string(rank) + "_");
    Taskflow<int> potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);
    Taskflow<int3> accu(&tp, verb);

    /*
    DepsLogger dlog(1000000);
    Logger log(1000000);
    tp.set_logger(&log);
    comm.set_logger(&log);
    */

    // Create active message
    auto am_trsm = comm.make_active_msg( 
            [&](view<double> &Lkk, int& k, view<int>& is) {
                *blocs[k+k*nb] = Map<MatrixXd>(Lkk.data(), n, n);
                for(auto& i: is) {
                    trsm.fulfill_promise({k,i});
                }
            });

        // Sends a panel bloc and trigger multiple gemms
    auto am_gemm = comm.make_active_msg(
        [&](view<double> &Lij, int& i, int& k, view<int2>& ijs) {
            *blocs[i+k*nb] = Map<MatrixXd>(Lij.data(), n, n);
            for(auto& ij: ijs) {
                gemm.fulfill_promise({k,ij[0],ij[1]});
            }
        });

    // Define the task flow
    potrf.set_task([&](int k) {
          timer t1 = wctime();
          LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, blocs[k+k*nb]->data(), n);
          timer t2 = wctime();
          potrf_us_t += 1e6 * elapsed(t1, t2);
          //potrf_us_t += 1;
          //printf("Running POTRF %d on rank %d, %d, %d\n", k, rank_3d[0], rank_3d[1], rank_3d[2]);
      })
        .set_fulfill([&](int k) {
            vector<vector<int>> fulfill(n_ranks);
            for (int i=k+1; i<nb; i++) {
                fulfill[rank3d21(i,k,k)].push_back(i);
                
            }
            
            for (int r = 0; r<n_ranks; r++) // Looping through all outgoing dependency edges
            {
                if (rank == r) {
                    for (auto& i: fulfill[r]) {
                        trsm.fulfill_promise({k,i});
                        //printf("Potrf %d fulfilling local Trsm (%d, %d) on rank %d\n", k, k, i, comm_rank());
                    }
                }
                else {
                    //cout<<"Sending data from "<<rank<<" to "<<r<<"\n";
                    auto Ljjv = view<double>(blocs[k+k*nb]->data(), n*n);
                    auto isv = view<int>(fulfill[r].data(), fulfill[r].size());
                    am_trsm->send(r, Ljjv, k, isv);

                }

            }
        })
        .set_indegree([&](int k) {

            if (k==0) {
                return 1;
            }
            else if (k < q) {
                return k;
            }
            else {
                return q;
            }
        })
        .set_mapping([&](int k) {

            return (k % n_threads);
        })
        .set_binding([&](int k) {
            return false;

        })        
        .set_priority([&](int k) {
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 3.0;
            }
            if (priority==1) {
                return 9.0*(double)(nb-k)-1.0;
            }
            if (priority==2) {
                return 9.0*(double)(nb-k)-1.0+18*nb;
            }
            if (priority==3) {
                return 9.0*(double)(nb-k)-1.0+18*nb*nb;
            }
            else {
                return 3.0*(double)(nb-k);
            }
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "POTRF" + to_string(k) + "_" + to_string(rank);
        });



    trsm.set_task([&](int2 ki) {
        int k=ki[0];
        int i=ki[1];
        timer t1 = wctime();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, n, n, 1.0, blocs[k+k*nb]->data(),n, blocs[i+k*nb]->data(), n);
        timer t2 = wctime();
        trsm_us_t += 1e6 * elapsed(t1, t2);
        //trsm_us_t += 1;
        //printf("Running TRSM (%d, %d) on rank %d, %d, %d\n", i, k, rank_3d[0], rank_3d[1], rank_3d[2]);
        //cout<<(*blocs[i+k*nb])<<"\n";

      })
        .set_fulfill([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            vector<vector<int2>> fulfill(n_ranks);
            for (int j=k+1; j<nb; j++) {
                if (j<i) {
                    fulfill[rank3d21(i,j,k)].push_back({i,j});
                }
                else {
                    fulfill[rank3d21(j,i,k)].push_back({j,i});
                }
                
            }
            for (int r=0; r<n_ranks; r++)   // Looping through all outgoing dependency edges
            {

                if (r == rank) {
                    for (auto& ij : fulfill[r]) {
                        gemm.fulfill_promise({k,ij[0],ij[1]});
                        //printf("Trsm (%d, %d) fulfilling local Gemm (%d, %d, %d) on rank %d\n", k, i, k, ij[0], ij[1], comm_rank());
                    }
                }
                else {
                    //cout<<"Sending data from "<<rank<<" to "<<r<<"\n";
                    auto Lijv = view<double>(blocs[i+k*nb]->data(), n*n);
                    auto ijsv = view<int2>(fulfill[r].data(), fulfill[r].size());
                    am_gemm->send(r, Lijv, i, k, ijsv);

                }
                

            }
        })
        .set_indegree([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            if (k==0) {
                return 1;
            }
            else if (k < q) {
                return k+1;
            }
            else {
                return q+1;
            }
        })
        .set_mapping([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];

            return ((k*n+i) % n_threads);
        })
        .set_binding([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            return false;

        })
        .set_priority([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 2.0;
            }
            if (priority==1) {
                return 9.0*(double)(nb-k)-2.0;
            }
            if (priority==2) {
                return 9.0*(double)(nb-k)-2.0+9*nb;
            }
            if (priority==3) {
                return (double)(nb-i)+nb*(9.0*(double)(nb-k)-2.0)+9*nb*nb;
            }
            else {
                return 2.0*(double)(nb-i);
            }
        })
        .set_name([&](int2 ki) { // This is just for debugging and profiling
            int k=ki[0];
            int i=ki[1];
            return "TRSM" + to_string(k) + "_" + to_string(i) + "_" +to_string(rank);
        });

    auto am_accu = comm.make_active_msg([&](view<double>& Lijk, int& i, int& j, int& from) {
        std::unique_ptr<MatrixXd> Atmp;
        Atmp = make_unique<MatrixXd>(n, n);
        *Atmp =  Map<MatrixXd>(Lijk.data(), n, n);
        gemm_results[i+j*nb].to_accumulate[from] = move(Atmp);
        accu.fulfill_promise({from, i, j});
    });



    gemm.set_task([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2]; 
            timer t1 = wctime();           
            if (i==j) { 
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, -1.0, blocs[i+k*nb]->data(), n, 1.0, blocs[i+j*nb]->data(), n);
            }
            else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, -1.0,blocs[i+k*nb]->data(), n, blocs[j+k*nb]->data(), n, 1.0, blocs[i+j*nb]->data(), n);
            }
            timer t2 = wctime();
            //printf("Running GEMM (%d, %d, %d) on rank %d, %d, %d\n", i, j, k, rank_3d[0], rank_3d[1], rank_3d[2]);
            gemm_us_t += 1e6 * elapsed(t1,t2);
            //gemm_us_t += 1;
            

      })
        .set_fulfill([&](int3 kij) { 
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (k+q<=j-1) {
                gemm.fulfill_promise({k+q, i, j});
                //printf("Gemm (%d, %d, %d) fulfilling Gemm (%d , %d, %d) on rank %d\n", k, i, j, k+1, i, j, comm_rank());
            }
            else {
                int dest =rank3d21(i % q ,j % q, j % q);
                if (dest == rank) {
                    //printf("Gemm (%d, %d, %d) fulfilling ACCUMU (%d, %d, %d) on rank %d, %d, %d\n", k, i, j, rank_3d[2], i, j, rank_3d[0], rank_3d[1], rank_3d[2]);
                    auto Lij = view<double>(blocs[i+j*nb]->data(), n*n);
                    std::unique_ptr<MatrixXd> Atmp;
                    Atmp = make_unique<MatrixXd>(n, n);
                    *Atmp = MatrixXd::Zero(n,n);
                    gemm_results[i+j*nb].to_accumulate[rank_3d[2]] = move(Atmp);
                    accu.fulfill_promise({rank_3d[2], i, j});
                }

                else {
                    int kk = rank_3d[2];
                    auto Lij = view<double>(blocs[i+j*nb]->data(), n*n);
                    //printf("Gemm (%d, %d, %d) Sending ACCUMU (%d, %d, %d) to rank %d, %d, %d\n", k, i, j, rank_3d[2], i, j, i, j, j);
                    am_accu->send(dest, Lij, i, j, kk);
                }
            }
            

        })
        .set_indegree([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            int t=3;
            if ((k/q)==0) {
                t--;
            }
            if (i==j) {
                t--;
            }
            return t;
        })
        .set_mapping([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];

            return ((k*n*n+i+j*n)  % n_threads);
        })
        .set_binding([&](int3 kij) {
            return false;

        })
        .set_priority([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 1.0;
            }
            if (priority==1) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==2) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==3) {
                return (double)(nb-i)+nb*(9.0*nb-3.0*j-6.0*k-2.0);
            }
            else {
                return 1.0*(double)(nb-i);
            }

        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            return "GEMM" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
        });

    accu.set_task([&](int3 kij) {
            assert(rank3d21(kij[1],kij[2],kij[2]) == rank);
            int k=kij[0]; // Step (gemm's pivot)
            int i=kij[1]; // Row
            int j=kij[2]; // Col
            assert(j <= i);
            //printf("Running ACCU (%d, %d, %d) on rank %d, %d, %d\n", k, i, j, rank_3d[0], rank_3d[1], rank_3d[2]);
            //assert(k < j);
            std::unique_ptr<Eigen::MatrixXd> Atmp;
            {
                lock_guard<mutex> lock(gemm_results[i+j*nb].mtx);
                Atmp = move(gemm_results[i+j*nb].to_accumulate[k]);
                gemm_results[i+j*nb].to_accumulate.erase(k);
            }
            //cout<<(*blocs[i+j*nb])<<"\n";
            //cout<<(*Atmp)<<"\n";
            timer t_ = wctime();
            *blocs[i+j*nb] += (*Atmp);
            timer t__ = wctime();
            accu_us_t += 1e6 * elapsed(t_, t__);
            //printf("Running ACCU (%d, %d, %d) on rank %d, %d, %d\n", k, i, j, rank_3d[0], rank_3d[1], rank_3d[2]);
        })
        .set_fulfill([&](int3 kij) {
            assert(rank3d21(kij[1],kij[2],kij[2]) == rank);
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            assert(j <= i);
            if(i == j) {
                potrf.fulfill_promise(i);
            } else {
                trsm.fulfill_promise({j,i});
            }
        })
        .set_indegree([&](int3 kij) {
            if (rank3d21(kij[1] % q, kij[2] % q,kij[2] % q) != rank) {
                printf("ACCU %d, %d, %d and current rank %d, %d, %d", kij[0], kij[1], kij[2], rank_3d[0], rank_3d[1], rank_3d[2]);
            }
            assert(rank3d21(kij[1] % q, kij[2] % q,kij[2] % q) == rank);
            
            return 1;
        })
        .set_mapping([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];

            return ((k*n*n+i+j*n)  % n_threads);// IMPORTANT. Every (i,j) should map to a given fixed thread
        })
        .set_priority([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 1.0;
            }
            if (priority==1) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==2) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==3) {
                return (double)(nb-i)+nb*(9.0*nb-3.0*j-6.0*k-2.0);
            }
            else {
                return 1.0*(double)(nb-i);
            }

        })
        .set_binding([&](int3 kij) {
            assert(rank3d21(kij[1],kij[2],kij[2]) == rank);
            return true; // IMPORTANT
        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            assert(rank3d21(kij[1],kij[2],kij[2]) == rank);
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            return "ACCUMU" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
        });


        

    MPI_Barrier(MPI_COMM_WORLD);

    timer t0 = wctime();
    if (rank == 0){
        potrf.fulfill_promise(0);
    }

    tp.join();
    MPI_Barrier(MPI_COMM_WORLD);
    timer t1 = wctime();
    
    MPI_Status status;
    
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            if (jj<=ii)  {
            if (rank==0 && rank!=rank3d21(ii, jj, jj)) {
                MPI_Recv(blocs[ii+jj*nb]->data(), n*n, MPI_DOUBLE, rank3d21(ii, jj, jj), 0, MPI_COMM_WORLD, &status);
                }

            else if (rank==rank3d21(ii, jj, jj) && rank != 0) {
                MPI_Send(blocs[ii+jj*nb]->data(), n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
    
    
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            L.block(ii*n,jj*n,n,n)=*blocs[ii+jj*nb];
        }
    }
    auto L1=L.triangularView<Lower>();
    if (rank==0) {
        cout<<"Priority "<<priority<<", Elapsed time: "<<elapsed(t0,t1)<<endl;
    }
    /*
    printf("Potrf time: %e\n", potrf_us_t.load() * 1e-6);
    printf("Trsm time: %e\n", trsm_us_t.load() * 1e-6);
    printf("Gemm time: %e\n", gemm_us_t.load() * 1e-6);
    printf("Rank %d Total Computation Time: %e\n", rank, potrf_us_t.load() * 1e-6+trsm_us_t.load() * 1e-6+gemm_us_t.load() * 1e-6);
    */
    /*
    printf("Potrf Time: %f\n", potrf_us_t.load());
    printf("Trsm Time: %f\n", trsm_us_t.load());
    printf("Gemms Time: %f\n", gemm_us_t.load());
    printf("Rank %d Total Computation Time: %f\n", rank, trsm_us_t.load()+potrf_us_t.load()+gemm_us_t.load());
    */
    LLT<MatrixXd> lltOfA(A);
    MatrixXd TrueL= lltOfA.matrixL();
/*     if (rank==0) {
    cout<<"True L:\n";
    cout<<TrueL<<"\n";
    cout<<"L: \n";
    cout<<L;
    } */
    VectorXd x = VectorXd::Random(n * nb);
    VectorXd b = A*x;
    VectorXd bref = b;
    L1.solveInPlace(b);
    L1.transpose().solveInPlace(b);
    double error = (b - x).norm() / x.norm();
    if (rank == 0) {
        cout << "Error solve: " << error << endl;
    }
/*     std::ofstream logfile;
    string filename = "ttor_distributed_Priority_"+to_string(n)+"_"+to_string(nb)+"_"+ to_string(n_threads)+"_"+ to_string(n_ranks)+"_"+ to_string(priority)+".log."+to_string(rank);
    logfile.open(filename);
    logfile << log;
    logfile.close(); */

}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);    



    int n_threads = 2;
    int verb = 0; // Can be changed to vary the verbosity of the messages
    int n=1;
    int nb=2;
    int n_col=1;
    int n_row=1;
    int priority=0;


    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }
    

    if (argc >= 5) {
        n_threads=atoi(argv[3]);
        verb=atoi(argv[4]);
    }

    if (argc >= 7) {
        n_col=atoi(argv[5]);
        n_row=atoi(argv[6]);
    }

    if (argc >= 8) {
        priority=atoi(argv[7]);
    }


    cholesky(n_threads, verb, n, nb, n_col, n_row, priority);

    MPI_Finalize();
}