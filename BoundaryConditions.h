/*
 *  BoundaryConditions.h
 *  Created by Jeremy Riousset on 6/26/08.
 */

#ifndef BOUNDARYCONDITIONS_H
#define BOUNDARYCONDITIONS_H

#include "MpiFunctions.h"

class BC
{
public:
	static void Apply(int BBCtype, CMatrix3D& pphi, CMatrix3D& llocal_rho);
//	static void Update(bool iisFlashAccoutedForInBC, int BBCtype, CMatrix3D& pphi_cha, const double rrhoAmbMin, const double rrhoAmbMax, const ResGrid& dd, const SizeGrid& NN);
//	static void AddUniformE(bool uuniformE, const double& EEo, CMatrix3D& pphi, CMatrix3D& UUn, const SizeDomain& LL, const ResGrid& dd, const SizeGrid& NN);
};
#endif BOUNDARYCONDITIONS_H
