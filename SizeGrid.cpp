/*
 *  SizeGrid.cpp
 *  Created by Jeremy Riousset on 10/25/07.
 */

#include "SizeGrid.h"

/**************************************************************************************/
SizeGrid::SizeGrid(){x = 1; y = 1; z = 1;};

SizeGrid::SizeGrid(int xx=1, int yy=1, int zz=1)
{SizeGrid::init(xx,yy,zz);};

double SizeGrid::max()
{
	double mmax;
	double ttmp = (x >= y)*x + (x < y)*y;
	mmax = (ttmp>= z)*ttmp + (ttmp < z)*z;
	return mmax;
};

bool SizeGrid::init(int xx, int yy, int zz)
{
	x = xx; y = yy; z = zz;
	return true;
};

SizeGrid::~SizeGrid(){};
/**************************************************************************************/

