#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <thread>

//may return 0 when not able to detect
const auto processor_count = std::thread::hardware_concurrency();

/***  OMP ***/
void seedThreads(const size_t nThreads, unsigned int* seeds) {
    int my_thread_id;
    unsigned int seed;
    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();

        //create seed on thread using current time
        unsigned int seed = (unsigned) time(NULL);

        //munge the seed using our thread number so that each thread has its
        //own unique seed, therefore ensuring it will generate a different set of numbers
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);

        printf("Thread %d has seed %u\n", my_thread_id, seeds[my_thread_id]);

    }

}
/***  OMP ***/


int main(int argc, char* argv[])
{
    unsigned int nThreads = processor_count;

    srand(time(NULL));
    unsigned int seeds[nThreads];
    omp_set_num_threads(nThreads);
    seedThreads(nThreads, seeds);

    double x, y, pi;
    size_t a = 1;
    size_t in_circ = 0;
    const size_t N = (argc > 1) ? atoi(argv[1]) : 500000;
    size_t tid, seed;


    #pragma omp parallel reduction(+: in_circ) private(x, y, tid, seed)
    {
        tid = omp_get_thread_num();
        seed = seeds[tid];
        srand(seed);

        #pragma omp for
        for (size_t i = 0; i < N; ++i)
        {
            x = (double)rand() / RAND_MAX;
            y = (double)rand() / RAND_MAX;
            x += rand() % a;
            y += rand() % a;

            if (x * x + y * y < a * a)
                in_circ++;
        }
    }
    pi = ((double) in_circ / N) * 4;
    printf("Pi = %f\n", pi);
    return 0;
}