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

	if( argc < 3 ){
      cout << "Error in inputs!" << endl;
      cout << "Run with [ReachabilityInstance] " << endl;
      exit(1);
	}

   ReachabilityInstance* inst = new ReachabilityInstance(argv[1], argv[2]);
//	inst->polytopes[7]->print(); exit(1);
   //inst->print();

   solve(inst);

   return 0;

}


void solve(ReachabilityInstance* inst){

	int bigM = 1000.0;
	if (inst->_benchmark.compare("Ball") == 0)
		bigM = 1000.0;
	else if (inst->_benchmark.compare("Oscillator") == 0)
		bigM = 10000.0;
	else if (inst->_benchmark.compare("Tanks") == 0)
		bigM = 100000.0;
	else if (inst->_benchmark.compare("Buck") == 0)
		bigM = 1000.0;
	else if (inst->_benchmark.compare("Filtered") == 0)
		bigM = 1000.0;
	else if (inst->_benchmark.compare("ISS") == 0)
		bigM = 10.0;
	else if (inst->_benchmark.compare("Particle") == 0)
		bigM = 100.0;

   std::cout << "big M is " << bigM << std::endl;
   //int bigM = 1000.0; // Ball String
   // int bigM = 10000.00; // Damped Oscillator
   //int bigM = 100000.0; // Two Tanks
   //int bigM = 1000.0; // Buck Converter
   // int bigM = 1000.0; // Filtered Osc 32
    try{

      GRBEnv env = GRBEnv();
      GRBModel model(env);

      GRBVar* x;

      x = new GRBVar[inst->n_variables];

      for(int j = 0 ; j < inst->n_variables ; ++j){
         //x[j] = model.addVar(-100000.0, 1.0000,0.0,GRB_CONTINUOUS); // For each benchmark so far
         // x[j] = model.addVar(-1000.0, 10.0,0.0,GRB_CONTINUOUS); // Ball String
         // x[j] = model.addVar(-100000.0, 1.0,0.0,GRB_CONTINUOUS); // Damped Oscillator 
         x[j] = model.addVar(-1000000.0, 100.0, 0.0, GRB_CONTINUOUS); // ISS
         // x[j] = model.addVar(-100000.0, 10.0, 0.0, GRB_CONTINUOUS); // Oscillating Particle 3 dim
	 //x[j] = model.addVar(-1000000.0, 10.0,0.0,GRB_CONTINUOUS); // Two Tanks
         // x[j] = model.addVar(-1000000.0, 1.0000,0.0,GRB_CONTINUOUS); // Buck Converter
         //x[j] = model.addVar(-1000000.0, 100.0000,0.0,GRB_CONTINUOUS); // Filtered Osc 32
      }

      GRBVar* z; // selects polytope

      char varname[256];

      z = new GRBVar[inst->n_timesteps];
      for(int t = 0 ; t < inst->n_timesteps ; ++t){
         sprintf(varname,"z[%d]",t);
         z[t] = model.addVar(0.0,1.0,0.0,GRB_BINARY,varname);
      }

      model.update();

      //// Each item can be assigned to at most 1

      for(int t = 0 ; t < inst->n_timesteps ; ++t){

         for(int c = 0 ; c < inst->polytopes[t]->n_constraints ; ++c){

            /*if(c == 1){
               if(t == 7){
                  double value = 0.0;
                  value += -5.6315*inst->polytopes[t]->con_matrix[c][0];
                  value += 0.0749016*inst->polytopes[t]->con_matrix[c][1];
                  cout << "LHS: " << endl;
                  for(int j = 0 ; j < inst->n_variables ; ++j){
                     cout << "\t" << inst->polytopes[t]->con_matrix[c][j];
                  }

                  cout << endl;
                  cout << "RHS: " << inst->polytopes[t]->rhs[c] << endl;
                  cout << "Val: " << value << endl;
               }
            }*/

            GRBLinExpr con = 0.0;
            for(int j = 0 ; j < inst->n_variables ; ++j){
               con += inst->polytopes[t]->con_matrix[c][j] * x[j];
            }
            sprintf(varname,"pcon[%d][%d]",t,c);
            model.addConstr(con <= inst->polytopes[t]->rhs[c] + bigM*( 1-z[t] ) , varname);
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

      cout << "Bounds:\t" << model.get(GRB_DoubleAttr_ObjVal) << "\t" << model.get(GRB_DoubleAttr_ObjBound) << endl;

      for(int t = 0 ; t < inst->n_timesteps ; ++t){
         if(z[t].get(GRB_DoubleAttr_X) > 0.9 ){
            cout << "\t" << t+1 << endl;
         }
      }


      for(int j = 0 ; j < inst->n_variables ; ++j){
         cout << "\t" << j << ": " << x[j].get(GRB_DoubleAttr_X) << endl;
      }


      for(int t = 0 ; t < inst->n_timesteps ; ++t){
         cout << "\tt: " << t << "\t" << z[t].get(GRB_DoubleAttr_X) << endl;
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
