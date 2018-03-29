#pragma once

#include <vector>
#include <iostream>
#include <time.h>
#include <algorithm>

using namespace std;

double interpolate_2d(double a, double b, double c, double d, double x, double y);
inline double interpolate(double a, double b, double x);
void smoothen(vector<vector<double>>& perlinMap);

class __declspec(dllexport) Distribution
{
public:
	/**
	 * @param tiles list of tiles frequencies.
	 * The order defines the height of the terrain : the head of the list is the lowest one, the last is the highest.
	 * Typically : water, desert, plain, mountains.
	 * @param decorators list of decorators frequencies.
	 */
	Distribution(vector<double>& terrains, vector<double>& decorators);

	/**
	 * @param r value between 0 and 1 defining the height of the terrain.
	 * @return the tile index and the decorator index corresponding to this terrain.
	 * The tile value is picked according to the height, the decorator value is purely random.
	 */
	vector<vector<pair<int, int>>> createMap(vector<vector<double>>& perlinMap);

private:
	vector<double> terrains;
	vector<double> decorators;
};

// TODO: city placement suggestions

class __declspec(dllexport) PerlinMap
{
public:
	PerlinMap(int height, int width, int octaves, double persistance, Distribution* distr);
	int getTerrain(int x, int y);
	int getDecorator(int x, int y);
private:
	Distribution* distr;
	int width;
	int height;
	vector<vector<pair<int, int>>> cases;

	static vector<vector<double>> generateWhiteNoise(int width, int height);
	static vector<vector<double>> generatePerlinNoise(int width, int height, int octaveCount, double persistance);
};
