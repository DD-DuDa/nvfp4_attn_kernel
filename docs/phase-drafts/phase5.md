# Phase 5 Draft — final pure FP4-Q gate

Run clean complete grid with trusted metadata and production iteration counts. D0 better FA4, D3 per head-config/seqlen batch geomean <=0.5, D4 FP4-Q gate. Report all head configs, overall diagnostic and worst point. If any seqlen/head gate fails, use measured memory bandwidth/bytes roofline, identify remaining margin and stop for user; do not continue residual or relax gate.
