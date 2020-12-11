#include <iostream>
#include <math.h>
#include <string>
#include <vector>
using namespace std;
#include <omp.h>

int thrdsCount = 2;


double func(double x){
    return (1/pow(x,2))*(pow(sin(1/x),2));
}

double trapezoid_atomic(double a, double b, double n) {

    double step = (b-a)/n;
    double x = 0;
    double sum = 0;
       #pragma omp parallel for num_threads(thrdsCount) private(x)
        for(int i=1; i<n; i++){
            x = (1/pow(a+i*step,2))*(pow(sin(1/(a+i*step)),2));
            #pragma omp atomic
            sum += x;
        }
    return (func(a)/2+func(b)/2+sum)*step;
}

double trapezoid_critical(double a, double b, double n) {
    omp_lock_t lock;	
    omp_init_lock(&lock);
    double step = (b-a)/n;
    double x = 0;
    double sum = 0;
    #pragma omp parallel for num_threads(thrdsCount) private(x)
        for(int i=1; i<n; i++){
            x = func(a+i*step);
            #pragma omp critical
            {
            sum += x;
            }

        }
    return (func(a)/2+func(b)/2+sum)*step;
}

double trapezoid_lock(double a, double b, double n) {
    omp_lock_t lock;	
    omp_init_lock(&lock);
    double step = (b-a)/n;
    double x = 0;
    double sum = 0;
    #pragma omp parallel for num_threads(thrdsCount) private(x)
        for(int i=1; i<n; i++){
            x = func(a+i*step);
            omp_set_lock (&lock);
            sum += x;
            omp_unset_lock (&lock);	
        }
    omp_destroy_lock (&lock);
    return (func(a)/2+func(b)/2+sum)*step;
}

double trapezoid_reduction(double a, double b, double n) {

    double step = (b-a)/n;
    double x = 0;
    double sum = 0;
    #pragma omp parallel num_threads(thrdsCount) private(x)
       #pragma omp for reduction(+:sum)
        for(int i=1; i<n; i++){
            x = func(a+i*step);
            sum += x;
        }
    return (func(a)/2+func(b)/2+sum)*step;
}



double trapezoid_sequential(double a, double b, double n) {
    double step = (b-a)/n;
    double sum = 0;
    for(int i=1; i<n; i++){
        sum += func(a+i*step);
    }
    return (func(a)/2+func(b)/2+sum)*step;
}


int main() {
    
    double a = 0.01;
    double b = 0.1;
    int n = 128000;

    double start = omp_get_wtime();
    cout << "Sequential results:\n";
    cout << "Integral value: "  << trapezoid_sequential(a,b,n) << "\n";
    cout << "Time: " << omp_get_wtime()-start << "\n"; 

    start = omp_get_wtime();
    cout << "Atomic results:\n";
    cout << "Integral value: " << trapezoid_atomic(a,b,n) << "\n";
    cout << omp_get_wtime()-start << "\n"; 

    start = omp_get_wtime();
    cout << "Critical results:\n";
    cout << "Integral value: " << trapezoid_critical(a,b,n) << "\n";
    cout << "Time: " << omp_get_wtime()-start << "\n";  
    
    start = omp_get_wtime();
    cout << "Lock results:\n";
    cout << "Integral value: " << trapezoid_lock(a,b,n) << "\n";
    cout << "Time: " << omp_get_wtime()-start << "\n";  

    start = omp_get_wtime();
    cout << "Reduction results:\n";
    cout << "Integral value: " << trapezoid_reduction(a,b,n) << "\n";
    cout << "Time: " << omp_get_wtime()-start << "\n"; 

    return 0;
}
