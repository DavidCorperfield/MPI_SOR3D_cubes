/*
 *  Input.h
 *  Created by Jeremy Riousset on 11/19/07.
 */

#ifndef INPUT_H
#define INPUT_H
#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
#include <mpi.h>
#include "SorSolution.h"

/*GLOBAL VARIABLES DECLARATION*****************************************************/
class Var
{
public:
	/*GLOBAL VARIABLES*/
	static	SizeGrid				N;												// Number of discretization points
	static	SizeDomain				L;												// Dimensions of the simulation domain
	static	ResGrid					d;												// Lengths of the discretization-grid
	
	static	const double			epsilon;										// SOR precision
	static	const int				MaxStep;										// Allowed maximum number of point per SOR iteration
	
	static	double					z_gnd;											// _m Altitude of the ground plane
	static	double					Q,  Xq,Yq,Zq, Rq1,Rq2,Rq3;						// Charge center parameters:
																					// * Charge (Q), 
																					// * X-,Y-,Z-coordinates of a charge center (Xq,Yq,Zq),
																					// * 1st, 2nd and 3rd geometrical parameters (Rq1, Rq2, Rq3) 		
	static	Charge					C;												// Charge center	

	static	ListCharge				ChargeCfg;										// Table with all parameters of the charge configuation
	
	static	CMatrix3D				phi;											// _V	Total electric potential
	static	CMatrix3D				rho;											// _C/m3	total charge density
	static	CMatrix3D				Un;												// Map of occupied grid points

	/*SLICE 3-D MATRICES*/
	static	SizeGrid				local_N;
	static	CMatrix3D				local_phi;
	static	CMatrix3D				local_rho;
	static	CMatrix3D				local_Un;
	static	Charge					local_C;
	
	/*PSOR ALGORITHM*/
	static	SorSolution				SOR;
};
/**********************************************************************************/

#endif // INPUT_H
