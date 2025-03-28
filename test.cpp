#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#define SOFTENING 1e-9f

void randomizeBodies(float* x, float* y, float* z,
                 float* vx, float* vy, float* vz, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}
void bodyForce(float* x, float* y, float* z, float* vx, float* vy,
               float* vz, float dt, int start, int end, int n) {
    for (int i = start; i <= end; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        for (int j = 0; j < n; j++) {
            float dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }
        vx[i] += dt * Fx; vy[i] += dt * Fy; vz[i] += dt * Fz;
    }
}

int main(int argc, char* argv[])
{
    srand(0);
    int MyID, NumProc, ierror, h, m, start = 0, end = 0, nBodies = 40960;
        if (argc > 1) nBodies = atoi(argv[1]);
    double tstart; double tfinish;
    float* X, * Y, * Z, * VX, * VY, * VZ;
    float local_center_x = 0., local_center_y = 0., local_center_z = 0.;
    float global_center_x = 0., global_center_y = 0., global_center_z = 0.;
    X = (float*)calloc(nBodies, sizeof(float)); Y = (float*)calloc(nBodies, sizeof(float));
    Z = (float*)calloc(nBodies, sizeof(float)); VX = (float*)calloc(nBodies, sizeof(float));
    VY = (float*)calloc(nBodies, sizeof(float)); VZ = (float*)calloc(nBodies, sizeof(float));
    float* Xk = (float*)calloc(nBodies, sizeof(float)); float* Yk = (float*)calloc(nBodies, sizeof(float));
    float* Zk = (float*)calloc(nBodies, sizeof(float));
    const float dt = 0.01f; const int nIters = 10;
    randomizeBodies(X, Y, Z, VX, VY, VZ, nBodies);
    MPI_Status status;
    ierror = MPI_Init(&argc, &argv);
    if (ierror != MPI_SUCCESS)
        printf("MPI initialization error!");
    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

    if (MyID == 0){
        printf("Num proc = %d\n", NumProc);
        printf("nBodies = %d\n", nBodies);
    }

      
    m = nBodies % NumProc; h = nBodies / NumProc;
    if (m != 0)
    {
        if (MyID < m)
            {h++; start = h * MyID; end = start + h - 1;}
        else
            {start = h * MyID + m; end = start + h - 1;}
    }
    else
        {start = h * MyID; end = start + h - 1;}
    for (int i = start; i <= end; i++)
    {
        int j = i - start;
        Xk[j] = X[i]; Yk[j] = Y[i]; Zk[j] = Z[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    tstart = MPI_Wtime();
    for (int i = 1; i <= nIters; i++)
    {
        bodyForce(X, Y, Z, VX, VY, VZ, dt, start, end, nBodies);
        for (int i = start; i <= end; i++) {
            int j = i - start;
            Xk[j] += VX[i] * dt; Yk[j] += VY[i] * dt; Zk[j] += VZ[i] * dt;
        }
        
        local_center_x = 0.;
        local_center_y = 0.;
        local_center_z = 0.;
        for (int i = 0; i <= end - start; ++i){
            local_center_x += Xk[i];
            local_center_y += Yk[i];
            local_center_z += Zk[i];
        }
        if (MyID == 0){
            global_center_x = 0.;
            global_center_y = 0.;
            global_center_z = 0.;
        }
        MPI_Reduce(&local_center_x, &global_center_x, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_center_y, &global_center_y, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_center_z, &global_center_z, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (MyID == 0) {
            global_center_x /= nBodies;
            global_center_y /= nBodies;
            global_center_z /= nBodies;
            printf("Iteration %d: Center Mass = (%.6f, %.6f, %.6f)\n", i, global_center_x, global_center_y, global_center_z);
        }

        MPI_Allgather(Xk, end - start + 1, MPI_FLOAT, X, end - start + 1, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(Yk, end - start + 1, MPI_FLOAT, Y, end - start + 1, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(Zk, end - start + 1, MPI_FLOAT, Z, end - start + 1, MPI_FLOAT, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    tfinish = MPI_Wtime() - tstart;

    if (MyID == 0){
        printf("MPI Time: %f\n", tfinish);
    }

    free(X); free(Y); free(Z); free(VX); free(VY); free(VZ);
    free(Xk); free(Yk); free(Zk);
    MPI_Finalize();
    return 0;
}

