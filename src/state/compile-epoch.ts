/** Monotone request version. Workers cannot interrupt JavaScript already
 * running, so completion is made cancellable by refusing stale results. */
export const compile = { epoch: 0 };
