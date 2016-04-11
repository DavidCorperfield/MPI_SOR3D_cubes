/*
 *  SizeDomain.cpp
 *  Created by Jeremy Riousset on 10/25/07.
 */

#include "SizeDomain.h"

/**************************************************************************************/
SizeDomain::SizeDomain(){x = 1; y = 1; z = 1;};
SizeDomain::SizeDomain(double xx=1,double yy=1,double zz=1)
{SizeDomain::init(xx,yy,zz);};

bool SizeDomain::init(double xx, double yy, double zz)
{
	x = xx; y = yy; z = zz;
	return true;
};

SizeDomain::~SizeDomain(){};
/**************************************************************************************/
