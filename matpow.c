#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <ctime>



void pow_sq(const size_t k, size_t *p2, size_t *k2) 
{
    p2[0] = 0; k2[0] = 1;

    if (k <= 1) return;

    while (k2[0] <= k) 
    {k2[0] <<= 1;++p2[0];}

    if (k2[0] > k)
    {k2[0] >>= 1; -p2[0];}
    return;
}


void matrix_multiply(const size_t N, const double *M1, const double *M2, double *res)
{
    double tmp[N * N];
    #pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for (size_t j = 0; j < N; ++j)
            tmp[i * N + j] = 0;
    #pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for (size_t j = 0; j < N; ++j)
            for(size_t k = 0; k < N; ++k)
				tmp[i * N + j] += M1[i * N + k] * M2[k * N + j];
    #pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for (size_t j = 0; j < N; ++j)
            res[i * N + j] = tmp[i * N + j];
    return;
}


void matrix_power(const size_t k, const size_t N, double *M)
{
    if (k == 0)
    {
    #pragma omp parallel for 
        for (size_t i = 0; i < N; ++i)   
            for (size_t j = 0; j < N; ++j)
                if (i != j)
                    M[i * N + j] = 0;
                else
                    M[i * N + j] = 1;
        return;
    }
    size_t k2 = 0, p2 = 0;
    double tmp[N * N];
    pow_sq(k, &p2, &k2);
    #pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for (size_t j = 0; j < N; ++j)
            tmp[i * N + j] = M[i * N + j];
    for (size_t i = 0; i < p2; ++i)
        matrix_multiply(N, &M[0], &M[0], &M[0]);
    for (size_t i = 0; i < (k - k2); ++i)
        matrix_multiply(N, &tmp[0], &M[0], &M[0]);
    return;
}


int main(int argc, char* argv[])
{
    int my_thread_id;
    unsigned int seed;

    const size_t nThreads = (argc > 3) ? atoi(argv[3]) : 4;

    srand(time(NULL));
    unsigned int seeds[nThreads];
    omp_set_num_threads(nThreads);

    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();
        unsigned int seed = (unsigned) time(NULL);
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
    }

    size_t tid;
    const size_t N = (argc > 1) ? atoi(argv[1]) : 3;
    const size_t POW = (argc > 2) ? atoi(argv[2]) : 3;
    const int MAX_VALUE = 10;
    double M[N * N];

    #pragma omp parallel private(tid, seed)
    {
        tid = omp_get_thread_num();
        seed = seeds[tid];
        srand(seed);
        #pragma omp for 
        for (size_t i = 0; i < N; ++i)
        {    
            for (size_t j = 0; j < N; ++j)
            { 
                M[i * N + j] = rand() % MAX_VALUE;
            }
        }
    }
    for (size_t i = 0; i < N; ++i)
    {    
        for (size_t j = 0; j < N; ++j)
            printf("%f ", M[i * N + j]); 
        printf("\n"); 
    }
    printf("\n\n");
    matrix_power(POW, N, &M[0]);
    for (size_t i = 0; i < N; ++i)
    {    
        for (size_t j = 0; j < N; ++j)
            printf("%f ", M[i * N + j]); 
        printf("\n"); 
    }
    printf("\n\n");
    return 0;
}