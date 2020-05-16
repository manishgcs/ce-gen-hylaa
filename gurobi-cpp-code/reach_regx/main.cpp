#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <list>
#include <cmath>
#include "math.h"
#include <vector>
#include <algorithm>
#include <iomanip>
#include "instance.h"
#include <vector>
///// THESE ARE FOR CPLEX
//#include <ilcplex/ilocplex.h>
//#include <ilcp/cpext.h>

#include "gurobi_c++.h"
using namespace std;

GRBEnv* env_main = NULL;


///// THIS REPLACES `using namespace std'
//ILOSTLBEGIN;

void solve(ReachabilityInstance* inst);

int main(int argc, char** argv){

//   GRBEnv* env_main;
   //env_main = new GRBEnv();
    // Checks that you are running with the correct number of command line arguments
//   env_main = new GRBEnv();

	if( argc < 2 ){
      cout << "Error in inputs!" << endl;
      cout << "Run with [ReachabilityInstance] " << endl;
      exit(1);
	}

   ReachabilityInstance* inst = new ReachabilityInstance(argv[1]);
//	inst->polytopes[7]->print(); exit(1);
   //inst->print();

   solve(inst);

   return 0;

}


void solve(ReachabilityInstance* inst){

   int bigM = 10000.00; // Damped Oscillator
   float epsilon = 0.00001;
   std:string reg_expr = "11111001111111111111111110";
   try{

      GRBEnv env = GRBEnv();
      GRBModel model(env);

      GRBVar* x;

      x = new GRBVar[inst->n_variables];

      for(int j = 0 ; j < inst->n_variables ; ++j){
         x[j] = model.addVar(-100000.0, 1.0,0.0,GRB_CONTINUOUS); // Damped Oscillator 
      }

      GRBVar* z; // selects polytope
      char varname[256];

      z = new GRBVar[inst->n_timesteps];
      for(int t = 0 ; t < inst->n_timesteps ; ++t){
	      sprintf(varname,"z[%d]",t);
	      z[t] = model.addVar(0.0,1.0,0.0,GRB_BINARY,varname);
      }

      model.update();
      
      std::vector< std::vector<GRBVar> > z_local;
      for(int t = 0; t < inst->n_timesteps ;++t){
	      vector<GRBVar> z_t;
         for(int c = 0 ; c < inst->polytopes[t]->n_constraints ; ++c){
	      sprintf(varname,"z_local[%d][%d]",t,c);
	      z_t.push_back(model.addVar(0.0, 1.0, 0.0, GRB_BINARY, varname));
	 }
	      z_local.push_back(z_t);
	}	
     
#if 0  
      std::vector< std::vector<GRBVar> >::const_iterator z_local_it = z_local.begin();
	for( ; z_local_it != z_local.end(); ++z_local_it) {	
		vector<GRBVar>::const_iterator z_t_it = z_local_it->begin();
		for( ; z_t_it != z_local_it->end(); ++z_t_it) {
			cout << z_t_it->get(GRB_DoubleAttr_X);
		}
	}
#endif
      for(int t = 0 ; t < inst->n_timesteps ; ++t){

	 GRBLinExpr zexpr = 0.0;

         for(int c = 0 ; c < inst->polytopes[t]->n_constraints ; ++c){

            GRBLinExpr con = 0.0;
            for(int j = 0 ; j < inst->n_variables ; ++j){
               con += inst->polytopes[t]->con_matrix[c][j] * x[j];
            }
            sprintf(varname,"pcon_1[%d][%d]",t,c);
            model.addConstr(con <= inst->polytopes[t]->rhs[c] + bigM*( 1-z_local[t][c] ) , varname);
            sprintf(varname,"pcon_2[%d][%d]",t,c);
            model.addConstr(con >= inst->polytopes[t]->rhs[c] + epsilon - bigM*( z_local[t][c] ) , varname);
            sprintf(varname,"zcon_1[%d][%d]",t,c);
            model.addConstr(z[t] <= z_local[t][c] , varname);
	    
	    zexpr += z_local[t][c];
         }
	 sprintf(varname,"zcon[%d]",t);
	 model.addConstr(zexpr <= z[t] + inst->polytopes[t]->n_constraints - 1, varname);
      	if(reg_expr[t] == '1') {
		model.addConstr(z[t] >= 1);
		model.addConstr(z[t] <= 1);
	}
	else
	{
		model.addConstr(z[t] >= 0);
		model.addConstr(z[t] <= 0);
	}
      }
      
      GRBLinExpr objective = 0.0;

      for(int t = 0 ; t < inst->n_timesteps ; ++t){
	      objective += z[t];
      }

      model.setObjective(objective,GRB_MAXIMIZE);

      model.getEnv().set(GRB_IntParam_Threads,1);
      model.getEnv().set(GRB_DoubleParam_TimeLimit,1800.0);
      //model.getEnv().set(GRB_IntParam_OutputFlag,0);

      model.write("reachability_model.lp");

      model.optimize();

      int status = model.get(GRB_IntAttr_Status);

      if ((status != GRB_OPTIMAL) || (status == GRB_INF_OR_UNBD) || (status == GRB_INFEASIBLE) || (status == GRB_UNBOUNDED))
	      cout << "Optimization was stopped with status " << status << endl;
      
      cout << "Bounds:\t" << model.get(GRB_DoubleAttr_ObjVal) << "\t" << model.get(GRB_DoubleAttr_ObjBound) << endl;

      for(int t = 0 ; t < inst->n_timesteps ; ++t){
         if(z[t].get(GRB_DoubleAttr_X) > 0.9 ){
            cout << "\t" << t+1 << endl;
         }
      }
      
      ofstream myfile;
      myfile.open("stats.txt",std::ios_base::app);
      for(int t = 0 ; t < inst->n_timesteps ; ++t){
         for(int c = 0 ; c < inst->polytopes[t]->n_constraints ; ++c){
         
	    myfile << "\t" << z_local[t][c].get(GRB_DoubleAttr_X) << endl;
         }
	 myfile << "\n";
      }


      for(int j = 0 ; j < inst->n_variables ; ++j){
         cout << "\t" << j << ": " << x[j].get(GRB_DoubleAttr_X) << endl;
      }


      for(int t = 0 ; t < inst->n_timesteps ; ++t){
         cout << "\tt: " << t << "\t" << z[t].get(GRB_DoubleAttr_X) << endl;
      }

        std::vector< std::vector<GRBVar> >::const_iterator z_local_it = z_local.begin();
	for( ; z_local_it != z_local.end(); ++z_local_it) {
		vector<GRBVar>::const_iterator z_t_it = z_local_it->begin();
		for( ; z_t_it != z_local_it->end(); ++z_t_it) {
			cout << "\t" << z_t_it->get(GRB_DoubleAttr_X);
		}
		cout << std::endl;	
	}


//      ofstream myfile;
//      myfile.open("stats.txt",std::ios_base::app);
//
//      myfile << inst->filename << "\t";
//      myfile << inst->n_variables << "\t";
//      myfile << inst->n_knapsacks << "\t";
//      myfile << solution_technique << "\t";
//      myfile << model.get(GRB_IntAttr_Status) << "\t";
//      myfile << model.get(GRB_DoubleAttr_ObjVal) << "\t" ;
//      myfile << model.get(GRB_DoubleAttr_ObjBound) << "\t" ;
//      myfile << model.get(GRB_DoubleAttr_Runtime) << "\t";
//      myfile << model.get(GRB_DoubleAttr_NodeCount) << "\t";
//      myfile << endl;
//
//      myfile.close();


   }catch(GRBException e){

      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
      exit(1);

   }

   catch(...) {
      cout << "Exception during optimization" << endl;
      exit(1);
   }
}
