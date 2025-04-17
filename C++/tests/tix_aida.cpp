/* Explanation method TIX for HiCs datasets */

#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/time.h>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <bits/stdc++.h>
#include <omp.h>

#include "isolation_formulas.h"
#include "aida_class.h"
#include "distance_metrics.h"
#include "io_module.h"

using namespace std;

int main(int argc, char** argv){
	struct timeval start, end;   // Time profiling 
	string tname = "example_cross_1000_50";
	string fname = "../synthetic_data/"+tname;
	string fnum  = fname+"_num.dat";
	ifstream foutliers(fname+"_outliers.dat");	
	
	int n, nFnum, nFnom = 1, n_outliers;
	int N = 100;  // Number of random subsamples
	
	// Load numerical features
	double *Xnum = NULL;
	read_data(Xnum,n,nFnum,fnum);
	
	// Load nominal features (or use array of 0's)
	int *Xnom = new int[n*nFnom];
	for(int i=0;i<n;i++){
		Xnom[i] = 0;
	}
	// Uncomment to use provided file with nominal features. In that case, also comment previous
	// declaration of Xnom.
//  int *Xnom = NULL;
//  read_data(Xnom,n,nFnom,fname+"_nom.dat");	
	
	/* Load outlier indexes */
	foutliers>>n_outliers;	
	int *outliers = new int[n_outliers];
	for(int i=0;i<n_outliers;i++){
		foutliers>>outliers[i];
	}
	foutliers.close();
	
	omp_set_num_threads(6);
	string score_type = "variance";
	string dist_type  = "manhattan";	
  string file_num   = (argc>1)?string("_")+argv[1]:"";
  double alpha_min = 1., alpha_max = 1.;
	
  // Train AIDA
	gettimeofday(&start, NULL);
	AIDA aida(N,"aom",score_type,alpha_min,alpha_max,dist_type);
	aida.fit(n,nFnum,Xnum,nFnom,Xnom,50,512,nFnum,nFnum,nFnom,nFnom);
	gettimeofday(&end, NULL);
	
	float delta = ((end.tv_sec-start.tv_sec)*1e6+end.tv_usec-start.tv_usec)/1.e6;
	
	cout<<"AIDA training time: "<<delta<<endl;  
	
  
	delta = 0.;
	ostringstream ref_factor_str;
	double ref_factor = 2.;
	int dim_ref = 10, niter_ref = 1, niter_tix = 10;
	double Tmin = 0.01, Tmax = 0.015;
	ref_factor_str<<setprecision(3)<<ref_factor;
  double *score_dim  = new double[nFnum];
  
	ofstream fex("../results/"+tname+"_TIX"+ref_factor_str.str()+file_num+".dat");
  fex<<n_outliers<<" "<<endl;
  for(int i=0;i<n_outliers;i++){
    aida.tix(score_dim,&Xnum[outliers[i]*nFnum],&Xnom[outliers[i]*nFnom],ref_factor,dim_ref,
             niter_ref,niter_tix,50*nFnum,Tmin,Tmax);
    
    // Write results in output file
		fex<<outliers[i]<<" ";
		for(int j=0;j<nFnum-1;j++){
			fex<<score_dim[j]<<" ";
		}
		fex<<score_dim[nFnum-1]<<endl;
  }
	fex.close();  
}
