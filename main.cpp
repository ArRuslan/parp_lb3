#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <windows.h>

void task_3_calc_pi_leibniz(int iterations) {
	double sum = 0;
	double sign = 1;
	for(int i = 0; i < iterations; i++) {
		sum += sign / (i * 2 + 1);
		sign *= -1;
	}

	std::cout << std::fixed << std::setprecision(36) << 4. * sum << "\n";
}

void task_3_calc_pi_numerical_integration_midpoint_singlethread(int iterations) {
	double sum = 0, step = 1. / iterations;
	for (int i = 0; i < iterations; i++) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	std::cout << std::fixed << std::setprecision(36) << step * sum << "\n";
}

void task_8_calc_pi_numerical_integration_midpoint_omp(int iterations) {
	int threads = omp_get_max_threads();
	omp_set_num_threads(threads);

	int i;
	double x;
	double sum = 0, step = 1. / iterations;

	#pragma omp parallel for reduction(+:sum) private(x)
	for (i = 0; i < iterations; i++) {
		x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}

	std::cout << std::fixed << std::setprecision(36) << step * sum << "\n";
}

struct ThreadData {
	int start;
	int end;
	double step;
	double sum;
};

uint32_t WINAPI task_6_calc_pi_numerical_integration_midpoint_win_thread(void* param) {
	auto* data = static_cast<ThreadData*>(param);
	double x;
	double local_sum = 0;

	for (int i = data->start; i < data->end; i++) {
		x = (i + 0.5) * data->step;
		local_sum += 4.0 / (1.0 + x * x);
	}

	data->sum = local_sum;

	return 0;
}

void task_6_calc_pi_numerical_integration_midpoint_win(int iterations) {
	const int num_threads = omp_get_max_threads();
	const double step = 1. / iterations;
	const int chunk_size = iterations / num_threads;
	double sum = 0;

	void** threads = new void*[num_threads];
	auto* thread_data = new ThreadData[num_threads];

	for (int t = 0; t < num_threads; t++) {
		thread_data[t].start = t * chunk_size;
		thread_data[t].end = (t == num_threads - 1) ? iterations : (t + 1) * chunk_size;
		thread_data[t].step = step;
		thread_data[t].sum = 0;
		threads[t] = CreateThread(nullptr, 0, task_6_calc_pi_numerical_integration_midpoint_win_thread, &thread_data[t], 0, nullptr);
	}

	WaitForMultipleObjects(num_threads, threads, true, INFINITE);

	for (int t = 0; t < num_threads; t++) {
		sum += thread_data[t].sum;
	}

	for (int t = 0; t < num_threads; t++) {
		CloseHandle(threads[t]);
	}

	delete[] threads;
	delete[] thread_data;

	std::cout << std::fixed << std::setprecision(36) << step * sum << "\n";
}

int main() {
#ifdef _OPENMP
	printf ("_OPENMP Defined\n");
#else
	printf("_OPENMP UnDefined\n");
#endif

	// Task 1, 2
	SYSTEM_INFO si;
	GetSystemInfo(&si);

	std::cout << "win cpu cores: " << si.dwNumberOfProcessors << "\n";
	std::cout << "omp cpu cores: " << omp_get_max_threads() << "\n";

	std::cout << std::fixed << std::setprecision(36) << M_PIl << "\n";

	// Task 3, 4, 5?
	double start_time = omp_get_wtime();
	task_3_calc_pi_leibniz(1000000000);
	double time_taken = omp_get_wtime() - start_time;
	std::cout << "Leibniz single-thread took " << std::fixed << std::setprecision(3) << time_taken << " seconds\n";

	start_time = omp_get_wtime();
	task_3_calc_pi_numerical_integration_midpoint_singlethread(1000000000);
	time_taken = omp_get_wtime() - start_time;
	std::cout << "Numerical integration midpoint single-thread took " << std::fixed << std::setprecision(3) << time_taken << " seconds\n";

	// Task 6, 7
	start_time = omp_get_wtime();
	task_6_calc_pi_numerical_integration_midpoint_win(1000000000);
	time_taken = omp_get_wtime() - start_time;
	std::cout << "Numerical integration midpoint WIN multi-thread took " << std::fixed << std::setprecision(3) << time_taken << " seconds\n";


	// Task 8, 9
	start_time = omp_get_wtime();
	task_8_calc_pi_numerical_integration_midpoint_omp(1000000000);
	time_taken = omp_get_wtime() - start_time;
	std::cout << "Numerical integration midpoint OMP multi-thread took " << std::fixed << std::setprecision(3) << time_taken << " seconds\n";

    return 0;
}
