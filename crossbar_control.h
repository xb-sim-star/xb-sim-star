//
// Created by author on 2019/3/21.
//

#ifndef _CROSSBAR_CONTROL_H
#define _CROSSBAR_CONTROL_H

#include "systemc.h"
#include "config.h"
#include<sys/time.h>

#ifdef USE_GPU
  #include "crossbar_cuda.h"
#else
  #include "crossbar.h"
#endif


using namespace std;

// crossbar computation controller

SC_MODULE(cb_control) {
    sc_in<bool> clock_2; // for crossbar computation

    void crossbar_compute() {

        entire_cb.run();
    }

    SC_CTOR(cb_control) {
        cout << "init entire crossbar..." << endl;
        entire_cb.init();

        SC_METHOD(crossbar_compute);
        sensitive << clock_2.neg();
        dont_initialize();
    }

    ~cb_control(){

    }
};


#endif //_CROSSBAR_CONTROL_H
