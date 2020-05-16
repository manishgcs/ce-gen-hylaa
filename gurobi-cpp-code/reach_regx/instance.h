#pragma once
#ifndef INSTANCE_HPP_
#define INSTANCE_HPP_

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

using namespace std;

struct Polytope{

    int n_constraints;
    vector<vector<double > >    con_matrix;
    vector<double >             rhs;

    void                        print();

    Polytope(){

    }

};

struct ReachabilityInstance{

    const char* filename;
    void read();
    void print();

    int n_variables;
    int n_timesteps;

    vector<Polytope* > polytopes;

    ReachabilityInstance(const char* _filename){
        n_variables = -1;
        n_timesteps = -1;
        filename = _filename;
        read();
    }

};



#endif
