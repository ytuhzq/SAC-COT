#pragma once
#include <boost/random.hpp>

namespace sac_cot
{
	namespace utils
	{	
		/**
		 * Random number generation
		 */
		void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx);
		void Rand(int seed, int scale, int& output1, int& output2, int& output3);
		void Rand(int seed, int scale, int& output1, int& output2);
		void Rand(int seed, int scale, int& output);

}
}
