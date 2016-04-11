/*
 *  Input.cpp
 *  Created by Jeremy Riousset on 11/19/07.
 */

#include "Input.h"


/*Variables Declaration************************************************************/
SizeGrid				Var::N;														// Number of discretization points
SizeDomain				Var::L;														// Dimensions of the simulation domain
ResGrid					Var::d;														// Lengths of the discretization-grid

const double			Var::epsilon		= 1e-10;								// SOR precision
const int				Var::MaxStep		= 7500;									// Allowed maximum number of point per SOR iteration

double					Var::z_gnd;													// _m Altitude of the ground plane
double					Var::Q,Var::Xq,Var::Yq,Var::Zq, Var::Rq1,Var::Rq2,Var::Rq3; // Charge center parameters:
																					// * Charge (Q), 
																					// * X-,Y-,Z-coordinates of a charge center (Xq,Yq,Zq),
																					// * 1st, 2nd and 3rd geometrical parameters (Rq1, Rq2, Rq3) 		
Charge					Var::C;														// Charge center	

ListCharge				Var::ChargeCfg;												// Table with all parameters of the charge configuation

CMatrix3D				Var::phi;													// _V	Total electric potential
CMatrix3D				Var::rho;													// _C/m3	total charge density
CMatrix3D				Var::Un;													// Map of occupied grid points

/*SLICE 3-D MATRICES*/
SizeGrid				Var::local_N;												// Size of local matrices
CMatrix3D				Var::local_phi;												// Scattered potential matrix
CMatrix3D				Var::local_rho;												// Scattered charge density matrix
CMatrix3D				Var::local_Un;												// Scattered matrix of interior boundary points
Charge					Var::local_C;												// Scattered charge

/*PSOR ALGORITHM*/
SorSolution				Var::SOR;													// Coefficient of SOR solver
/**********************************************************************************/

