/*
 *  ResGrid.h
 *  Created by Jeremy Riousset on 10/25/07.
 */

#ifndef RESGRID_H
#define RESGRID_H

#include <cmath>
#include "SizeDomain.h"
#include "SizeGrid.h"

/**************************************************************************************/
class ResGrid
{
public:
	double x,y,z,xy,yz,xz,xyz;
	ResGrid();									// Default Constructor
	ResGrid(SizeDomain L, SizeGrid N);			// Surcharged constructor
	bool init(SizeDomain L, SizeGrid N);			// Initiation after declaration
	~ResGrid();									// Destructor
};
/**************************************************************************************/

#endif // RESGRID_H
