/*
 *  SorSolution.h
 *  Created by Jeremy Riousset on 11/19/07.
 */

#ifndef SORSOLUTION_H
#define SORSOLUTION_H

#include "Sources.h"
#include "MpiInput.h"

/**************************************************************************************/
enum SourceType {ChargeDistribution, PotentialDistribution};
enum PointsColor {black, red};
// Allowed type of sources to use SOR solver
/**************************************************************************************/

/**************************************************************************************/
class SorSolution
{
private:
	double epsilon;																		// Maximum tolerable error
	int MaxStep;																		// Maximum number of iterations
	double a,b,c,d,e,f,g;																// Laplace's Equation coefficients
	CMatrix3D h;																		// Laplace's Equation coefficients (cont.)
	SourceType type;																	// Type of source
	double ErrDen;																		// Normalisation of the error
	
public:	
	SorSolution(){};																	// Default constructor
	SorSolution(CMatrix3D& pphi, double eepsilon,int MMaxStep, ResGrid dd,SizeGrid NN, Charge& CC,		const	CMatrix3D& UUn);	
																						// Constructor surcharge
	SorSolution(CMatrix3D& pphi, double eepsilon,int MMaxStep, ResGrid dd,SizeGrid NN, Potential& PP,			CMatrix3D& UUn);	
																						// Constructor surcharge
	~SorSolution(){};																	// Destructor
	void init(CMatrix3D& pphi, double eepsilon,int MMaxStep, ResGrid dd,SizeGrid NN, Charge& CC,	const	CMatrix3D& UUn);
	void init(CMatrix3D& pphi, double eepsilon,int MMaxStep, ResGrid dd,SizeGrid NN, Potential& PP,			CMatrix3D& UUn);
	void Update_GhostPlanes(CMatrix3D& MM, SizeGrid local_N, PointsColor CC);			// Exchange rows
	void Solve(ResGrid dd, SizeGrid NN, const CMatrix3D& UUn, CMatrix3D& pphi);			// Solve solution using SOR method
};
/**************************************************************************************/

#endif // SORSOLUTION_H
