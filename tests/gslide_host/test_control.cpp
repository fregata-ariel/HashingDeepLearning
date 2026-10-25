#include "TrainingControl.h"
#include <iostream>
#include <stdexcept>
#include <string>
int main() {
  int cases = 0;
  for (int layers : {0, 2})
    for (bool rebuild : {false, true})
      for (bool reshuffle : {false, true}) {
        std::string actual;
        gslide::TrainingOperations op;
        op.hidden_layers = layers;
        op.transfer = [&] { actual += 'T'; };
        op.hidden_forward = [&](int i) { actual += char('a' + i); };
        op.output_forward = [&] { actual += 'F'; };
        op.reset_hidden_deltas = [&] { actual += 'R'; };
        op.output_backward = [&] { actual += 'B'; };
        op.hidden_backward = [&](int i) { actual += char('c' + i); };
        op.hidden_update = [&](int i) { actual += char('e' + i); };
        op.output_update = [&] { actual += 'U'; };
        op.rebuild_index = [&](bool shuffle) { actual += shuffle ? 'S' : 'I'; };
        op.complete = [&] { actual += 'C'; };
        gslide::train_step(op, rebuild, reshuffle);
        std::string expected = layers == 2 ? "TabFRBdcefU" : "TFRBU";
        if (rebuild || reshuffle)
          expected += reshuffle ? 'S' : 'I';
        expected += 'C';
        if (actual != expected)
          throw std::runtime_error("training control order");
        ++cases;
      }
  std::cout << "GSLIDE_HOST_UNIT suite=control cases=" << cases
            << " sycl_required=false\n";
}
