#include "instance.h"
///// THESE ARE FOR CPLEX
//#include <ilcplex/ilocplex.h>
//#include <ilcp/cpext.h>

#include "math.h"

void ReachabilityInstance::print(){
   cout << "Printing instance .. " << endl;
   cout << "N timesteps: " << n_timesteps << endl;
   cout << "N variables: " << n_variables << endl;
   for(int t = 0 ; t < (int)polytopes.size() ; ++t){
      polytopes[t]->print();
   }
};

void Polytope::print(){
   cout << "Print polytope with " << n_constraints << " constraints .. " << endl;
   for(int i = 0 ; i < n_constraints ; ++i){
      for(int j = 0 ; j < con_matrix[i].size() ; ++j){
         cout << con_matrix[i][j] << "\t";
      }
      cout << rhs[i];
      cout << endl;
   }

};

void ReachabilityInstance::read(){

   int reader_int = 0;
   double reader_double = 0;

   ifstream inputfile;
   inputfile.open(_filename);

   inputfile >> n_variables;
   inputfile >> n_timesteps;

   Polytope* polytope;

   for(int t = 0 ; t < n_timesteps ; ++t){
      polytope = new Polytope();
      inputfile >> reader_int;
      polytope->n_constraints = reader_int;
      polytope->con_matrix.resize(polytope->n_constraints);
      polytope->rhs.resize(polytope->n_constraints,0.0);

      for(int i = 0 ; i < polytope->n_constraints ; ++i){
         polytope->con_matrix[i].resize(n_variables,0.0);
         inputfile >> reader_int;
         int n_vars_in_con = reader_int;

         for(int j = 0 ; j < n_vars_in_con ; ++j){
            inputfile >> reader_int;
            int var = reader_int - 1;
            inputfile >> reader_double;
            double coef = reader_double;
            polytope->con_matrix[i][var] = coef;
         }
         inputfile >> reader_double;
         polytope->rhs[i] = reader_double;
      }
      polytopes.push_back(polytope);

   }

};



