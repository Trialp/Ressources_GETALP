// INSAWrapper.h
#pragma once

#include <iostream>
#include <vector>
#include "INSALib.h"

using namespace System;
using namespace std;

namespace INSAWrapper {
	public ref class PerlinMapWrapper
	{
	public:
		PerlinMapWrapper(int size, int octaves, double persistance, array<double>^ tiles, array<double>^ decorators)
		{
			vector<double> tiles_c;
			vector<double> decorators_c;

			for (int i = 0; i < tiles->Length; i++)
			{
				double f = tiles[i];
				tiles_c.push_back(f);
			}

			for (int i = 0; i < decorators->Length; i++)
			{
				double f = decorators[i];
				decorators_c.push_back(f);
			}

			Distribution *distr = new Distribution(tiles_c, decorators_c);

			map = new PerlinMap(size, size, octaves, persistance, distr);
		}

		~PerlinMapWrapper()
		{
			delete map;
		}

		int GetTerrain(int x, int y)
		{
			return map->getTerrain(x, y);
		}

		int GetDecorator(int x, int y)
		{
			return map->getDecorator(x, y);
		}
	private:
		PerlinMap *map;
	};
}
