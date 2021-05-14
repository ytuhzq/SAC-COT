#include"utils.h"

namespace sac_cot
{
	namespace utils
	{
		/**
		* Random number generation
		*/
		void boost_rand(int seed, int start, int end, int rand_num, std::vector<int>& idx)
		{
			boost::mt19937 engine(seed);
			boost::uniform_int<> distribution(start, end);
			boost::variate_generator<boost::mt19937, boost::uniform_int<> > myrandom(engine, distribution);
			for (int i = 0; i < rand_num; i++)
				idx.push_back(myrandom());
		}
		void Rand(int seed, int scale, int& output1, int& output2, int& output3)
		{
			std::vector<int> result;
			int start = 0;
			int end = scale - 1;
			boost_rand(seed, start, end, scale, result);
			output1 = result[0];
			output2 = result[1];
			output3 = result[2];
		}
		void Rand(int seed, int scale, int& output1, int& output2)
		{
			std::vector<int> result;
			int start = 0;
			int end = scale - 1;
			boost_rand(seed, start, end, scale, result);
			output1 = result[0];
			output2 = result[1];
		}
		void Rand(int seed, int scale, int& output)
		{
			std::vector<int> result;
			int start = 0;
			int end = scale - 1;
			boost_rand(seed, start, end, scale, result);
			output = result[0];
		}
	}
}
