#include <stdio.h>
#include <math.h>

int main(void) {
    int a[4][4] = {
        {1, 2, 0, 1},
        {3, 1, 2, 2},
        {0, 1, 3, 1},
        {2, 2, 1, 0}
    };

    double b[2][2] = {
        {1, -1},
        {0,  1}
    };

    double c[3][3] = {0};
    double p[2][2] = {0};

    int maxRow[2][2], maxCol[2][2];
    int i, j, x, y, k;
// crosscorelatin
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (x = 0; x < 2; x++) {
                for (y = 0; y < 2; y++) {
                    c[i][j] += a[i+x][j+y] * b[x][y];
                }
            }
        }
    }

    printf("\nCross-correlation:\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("%.2f ", c[i][j]);
        }
        printf("\n");
    }

    // 2. ReLU
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (c[i][j] < 0) {
                c[i][j] = 0;
            }
        }
    }

    printf("\nAfter ReLU:\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("%.2f ", c[i][j]);
        }
        printf("\n");
    }
    // 3. Maxpooling
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            p[i][j] = c[i][j];
            maxRow[i][j] = i;
            maxCol[i][j] = j;
            for (x = 0; x < 2; x++) {
                for (y = 0; y < 2; y++) {
                    if (c[i+x][j+y] > p[i][j]) {
                        p[i][j] = c[i+x][j+y];
                        maxRow[i][j] = i+x;
                        maxCol[i][j] = j+y;
                    }
                }
            }
        }
    }
    printf("\n After Maxpooling:\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            printf("%.2f ", p[i][j]);
        }
        printf("\n");
    }

    // 4. Flatten
    double f[4];
    k = 0;

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            f[k++] = p[i][j];
        }
    }

    printf("\nAfter Flattening:\n");
    for (i = 0; i < 4; i++) {
        printf("%.2f ", f[i]);
    }
    printf("\n");

    // 5. FCNN
    double w[2][4] = {
        {1, 0, 1, 0},
        {0, 1, 0, 1}
    };

    double bias[2] = {0, 0};
    double z[2] = {0};

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            z[i] += f[j] * w[i][j];
        }
        z[i] += bias[i];
    }

    printf("\nlogit :\n");
    for (i = 0; i < 2; i++) {
        printf("%.2f ", z[i]);
    }
    printf("\n");

    // 6. Softmax
    double soft[2];
    double sum = 0.0;
    double maxZ = (z[0] > z[1]) ? z[0] : z[1];

    for (i = 0; i < 2; i++) {
        soft[i] = exp(z[i] - maxZ);
        sum += soft[i];
    }

    for (i = 0; i < 2; i++) {
        soft[i] /= sum;
    }

    printf("\nSoftmax:\n");
    for (i = 0; i < 2; i++) {
        printf("%.4f ", soft[i]);
    }
    printf("\n");

    // 7. Cross-entropy loss
    int target[2] = {0, 1};
    double loss = 0.0;

    for (i = 0; i < 2; i++) {
        loss -= target[i] * log(soft[i]);
    }

    printf("\nLoss = %.4f\n", loss);

    // BACKWARD PASS 

    // 8. Gradient at logits: dz = softmax - target
    // dz = dow L/dowzi
    double dz[2];
    for (i = 0; i < 2; i++) {
        dz[i] = soft[i] - target[i];
    }

    printf("\nGradient values(dz):\n");
    for (i = 0; i < 2; i++) {
        printf("%.4f ", dz[i]);
    }
    printf("\n");

    // 9.Gardients of weight and bias 
    double dw[2][4];
    double db[2];
    for (i = 0; i < 2; i++) {
        db[i] = dz[i];
        for (j = 0; j < 4; j++) {
            dw[i][j] = dz[i] * f[j];
        }
    }
    printf("\nWeight gradients(dw):\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            printf("%.4f ", dw[i][j]);
        }
        printf("\n");
    }

    printf("\nBias gradients(db):\n");
    for (i = 0; i < 2; i++) {
        printf("%.4f ", db[i]);
    }
    printf("\n");

    // 10. Gradient of FCNN 
    double df[4] = {0};
    for (j = 0; j < 4; j++) {
        for (i = 0; i < 2; i++) {
            df[j] += dz[i] * w[i][j];
        }
    }
    printf("\nGradient of fcnn (df):\n");
    for (i = 0; i < 4; i++) {
        printf("%.4f ", df[i]);
    }
    printf("\n");

   // 1d vector to 2x2 matrix
    double dp[2][2];
    k = 0;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            dp[i][j] = df[k++];
        }
    }

    // 12. Maxpool backward
    double dc[3][3] = {0};

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            dc[maxRow[i][j]][maxCol[i][j]] += dp[i][j];
        }
    }

    printf("\nGradient after Maxpooling backward pass :\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("%.4f ", dc[i][j]);
        }
        printf("\n");
    }

    // 13. ReLU backward
    // c contains ReLU output,only positive values pass gradient
    double dcorr[3][3] = {0};

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (c[i][j] > 0) {
                dcorr[i][j] = dc[i][j];
            }
        }
    }

    printf("\nGradient after ReLU backward:\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("%.4f ", dcorr[i][j]);
        }
        printf("\n");
    }

    // 14.  kernel gradient
    double dkernel[2][2] = {0};

    for (x = 0; x < 2; x++) {
        for (y = 0; y < 2; y++) {
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    dkernel[x][y] += dcorr[i][j] * a[i+x][j+y];
                }
            }
        }
    }

    printf("\nKernel gradients:\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            printf("%.4f ", dkernel[i][j]);
        }
        printf("\n");
    }
    // Updating parameters

int epochs;
double lr;

printf("Number of epochs: ");
scanf("%d", &epochs);

printf("Learning rate: ");
scanf("%lf", &lr);

// Update FCNN weights and bias

for (i = 0; i < 2; i++) {

    bias[i] -= lr * db[i];

    for (j = 0; j < 4; j++) {
        w[i][j] -= lr * dw[i][j];
    }
}

// Update CNN kernel

for (x = 0; x < 2; x++) {
    for (y = 0; y < 2; y++) {
        b[x][y] -= lr * dkernel[x][y];
    }
}

printf("\nUpdated Kernel:\n");

for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
        printf("%.4f ", b[i][j]);
    }
    printf("\n");
}

printf("\nUpdated FCNN weights:\n");

for (i = 0; i < 2; i++) {
    for (j = 0; j < 4; j++) {
        printf("%.4f ", w[i][j]);
    }
    printf("\n");
}

printf("\nUpdated Bias:\n");

for (i = 0; i < 2; i++) {
    printf("%.4f ", bias[i]);
}

    return 0;
}