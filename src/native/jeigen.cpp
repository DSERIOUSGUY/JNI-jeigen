// Copyright Hugh Perkins 2012, hughperkins -at- gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <iostream>
#include <jni.h>
#include <android/log.h>

using namespace std;

#include "jeigen.h"

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "unsupported/Eigen/MatrixFunctions"
using namespace Eigen;

extern "C" {
    JNIEXPORT jint JNICALL Java_com_example_datacollection_jeigen_myNativeMethod(JNIEnv *env, jclass clazz, jint arg) {
        return arg * arg;
    }
}


/* use Map intead of this method
void valuesToMatrix( int rows, int cols, double *values, MatrixXd *pM ) {
    int i = 0;
    for( int c = 0; c < cols; c++ ) {
        for ( int r = 0; r < rows; r++ ) {
            (*pM)(r,c) = values[i];
            i++;
        }
    }
}*/

void matrixToValues( int rows, int cols, const MatrixXd *pM, double *values ) {
    int i = 0;
    for( int c = 0; c < cols; c++ ) {
        for ( int r = 0; r < rows; r++ ) {
            values[i] = (*pM)(r,c);
            i++;
        }
    }
}

const int RESULTS_SIZE = 1000;
void *data[RESULTS_SIZE];

int storeData_(void *m) {
    for( int i = 0; i < RESULTS_SIZE; i++ ) {
        if( data[i] == 0 ) {
            data[i] = m;
            return i;
        }
    }
    return 0;
}
SparseMatrix<double> *getSparseMatrix_(int handle ) {
    return (SparseMatrix<double> *)(data[handle]);
}

int print_log(string tag, string message){
//    const char *tag_ptr = tag.c_str();
    return __android_log_print(ANDROID_LOG_DEBUG, "test", "%s", message.c_str());
}

extern "C" {
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_init(JNIEnv *env) {
        for( int i = 0; i < RESULTS_SIZE; i++ ) {
            data[i] = 0;
        }
    }
    JNIEXPORT jint JNICALL Java_com_example_datacollection_jeigen_allocateSparseMatrix( JNIEnv *env, jclass clazz, jint numEntries, jint numRows, jint numCols, jintArray rows_arr, jintArray cols_arr, jdoubleArray values_arr ) {
        SparseMatrix<double> *pmat = new SparseMatrix<double>(numRows, numCols);
        pmat->reserve(numEntries);
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;

        jint* rows = env->GetIntArrayElements(rows_arr, JNI_FALSE);
        jint* cols = env->GetIntArrayElements(cols_arr,JNI_FALSE);
        jdouble* values = env->GetDoubleArrayElements(values_arr,JNI_FALSE);
//        print_log("test","hello2");



        for( int i = 0; i < numEntries; i++ ) {
            tripletList.push_back(T(rows[i],cols[i],values[i]));
        }
        pmat->setFromTriplets(tripletList.begin(), tripletList.end() );
        return storeData_(pmat);
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_getSparseMatrixStats( JNIEnv *env, jclass clazz, jint handle, jint* stats ) { // rows, cols, nonzero
        stats[0] = getSparseMatrix_(handle)->rows();
        stats[1] = getSparseMatrix_(handle)->cols();
        stats[2] = getSparseMatrix_(handle)->nonZeros();
    }
    JNIEXPORT jint JNICALL Java_com_example_datacollection_jeigen_getSparseMatrixNumEntries( int handle ) {
        return getSparseMatrix_(handle)->nonZeros();
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_getSparseMatrix( JNIEnv *env, jclass clazz, jint handle, jintArray rows_arr, jintArray cols_arr, jdoubleArray values_arr ) {
        SparseMatrix<double> *pmat = getSparseMatrix_(handle);
        int numEntries = pmat->nonZeros();
        int i = 0;

        jint* rows = env->GetIntArrayElements(rows_arr, JNI_FALSE);
        jint* cols = env->GetIntArrayElements(cols_arr,JNI_FALSE);
        jdouble* values = env->GetDoubleArrayElements(values_arr,JNI_FALSE);

        for (int k=0; k<pmat->outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(*pmat,k); it; ++it) {
                rows[i] = it.row();
                cols[i] = it.col();
                values[i] = it.value();
                i++;
            }
      }
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_freeSparseMatrix( JNIEnv *env, jclass clazz, jint handle ) {
        if( handle < 0 || handle >= RESULTS_SIZE ) {
            throw std::runtime_error("handle out of range");
        }
        delete (SparseMatrix<double> *)(data[handle] );
        data[handle] = 0;
    }
    // dummy operation to measure end to end latency
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_dense_dummy_op1( JNIEnv *env, jclass clazz, jint rows, jint cols, jdoubleArray afirst_arr, jdoubleArray aresult_arr ) {

        jdouble* afirst = env->GetDoubleArrayElements(afirst_arr, JNI_FALSE);
        jdouble* aresult = env->GetDoubleArrayElements(aresult_arr,JNI_FALSE);

        Map<MatrixXd> first(afirst,rows,cols);
        Map<MatrixXd> result(aresult,rows,cols);
    }
    // dummy operation to measure end to end latency
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_dense_dummy_op2( JNIEnv *env, jclass clazz, jint rows, jint middle, jint cols, jdoubleArray afirst_arr, jdoubleArray asecond_arr, jdoubleArray aresult_arr ) {

        jdouble* afirst = env->GetDoubleArrayElements(afirst_arr, JNI_FALSE);
        jdouble* aresult = env->GetDoubleArrayElements(aresult_arr,JNI_FALSE);
        jdouble* asecond = env->GetDoubleArrayElements(asecond_arr, JNI_FALSE);

        Map<MatrixXd>first(afirst,rows,middle);
        Map<MatrixXd>second(asecond,middle,cols);
        Map<MatrixXd>result(aresult,rows,cols);
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_dense_multiply( JNIEnv *env, jclass clazz, jint rows, jint middle, jint cols, jdoubleArray afirst_arr, jdoubleArray asecond_arr, jdoubleArray aresult_arr ) {

        jdouble* afirst = env->GetDoubleArrayElements(afirst_arr, JNI_FALSE);
        jdouble* aresult = env->GetDoubleArrayElements(aresult_arr,JNI_FALSE);
        jdouble* asecond = env->GetDoubleArrayElements(asecond_arr, JNI_FALSE);

        Map<MatrixXd>first(afirst,rows,middle);
        Map<MatrixXd>second(asecond,middle,cols);
        Map<MatrixXd>result(aresult,rows,cols);
        result = first * second;
    }
    JNIEXPORT jint JNICALL Java_com_example_datacollection_jeigen_sparse_multiply( JNIEnv *env, jclass clazz, jint rows, jint middle, jint cols,
        jint onehandle, jint twohandle ) {
        SparseMatrix<double> *presult = new SparseMatrix<double>(rows,cols);
        *presult = (*getSparseMatrix_(onehandle)) * (*getSparseMatrix_(twohandle));
        return storeData_(presult);
    }
    JNIEXPORT jint JNICALL Java_com_example_datacollection_jeigen_sparse_dummy_op2( JNIEnv *env, jclass clazz, jint rows, jint middle, jint cols,
        jint onehandle, jint twohandle, jint numResultColumns ) {
        SparseMatrix<double> *presult = new SparseMatrix<double>(rows,cols);
        presult->reserve(numResultColumns*rows);
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        for( int c = 0; c < numResultColumns; c++ ) {
            for( int r = 0; r < rows; r++ ) {
                tripletList.push_back(T(r,c,1));
            }
        }
        presult->setFromTriplets(tripletList.begin(), tripletList.end() );
        return storeData_(presult);
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_sparse_dense_multiply( JNIEnv *env, jclass clazz, jint rows, jint middle, jint cols, jint onehandle, jdoubleArray asecond_arr, jdoubleArray aresult_arr ) {

        jdouble* aresult = env->GetDoubleArrayElements(aresult_arr,JNI_FALSE);
        jdouble* asecond = env->GetDoubleArrayElements(asecond_arr, JNI_FALSE);

        Map<MatrixXd> second(asecond,middle,cols);
        Map<MatrixXd> result(aresult,rows,cols);
        result = (*getSparseMatrix_(onehandle)) * second;
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_dense_sparse_multiply( JNIEnv *env, jclass clazz, jint rows, jint middle, jint cols, jdoubleArray afirst_arr, jint twohandle, jdoubleArray aresult_arr ) {

        jdouble* afirst = env->GetDoubleArrayElements(afirst_arr,JNI_FALSE);
        jdouble* aresult = env->GetDoubleArrayElements(aresult_arr, JNI_FALSE);

        Map<MatrixXd> first(afirst, rows,middle);
        Map<MatrixXd> result(aresult,rows,cols);
        result =  first * (*getSparseMatrix_(twohandle));
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_ldlt_solve( JNIEnv *env, jclass clazz, jint arows, jint acols, jint bcols, jdoubleArray avalues_arr, jdoubleArray bvalues_arr, jdoubleArray xvalues_arr ) {

        jdouble* avalues = env->GetDoubleArrayElements(avalues_arr,JNI_FALSE);
        jdouble* bvalues = env->GetDoubleArrayElements(bvalues_arr, JNI_FALSE);
        jdouble* xvalues = env->GetDoubleArrayElements(xvalues_arr, JNI_FALSE);

        Map<MatrixXd> A(avalues,arows, acols);
        Map<MatrixXd> b(bvalues, arows, bcols);
        Map<MatrixXd> result(xvalues, acols, bcols);
        result = A.ldlt().solve(b);
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_fullpivhouseholderqr_solve( JNIEnv *env, jclass clazz, jint arows, jint acols, jint bcols, jdoubleArray avalues_arr, jdoubleArray bvalues_arr, jdoubleArray xvalues_arr ) {

        jdouble* avalues = env->GetDoubleArrayElements(avalues_arr,JNI_FALSE);
        jdouble* bvalues = env->GetDoubleArrayElements(bvalues_arr, JNI_FALSE);
        jdouble* xvalues = env->GetDoubleArrayElements(xvalues_arr, JNI_FALSE);

        Map<MatrixXd> A(avalues, arows, acols);
        Map<MatrixXd> b(bvalues, arows, bcols);
        Map<MatrixXd> result(xvalues, acols, bcols);
        result = A.fullPivHouseholderQr().solve(b);
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_svd_dense( JNIEnv *env, jclass clazz, jint n, jint p, jdoubleArray in_arr, jdoubleArray u_arr, jdoubleArray s_arr, jdoubleArray v_arr ) {

        jdouble* in = env->GetDoubleArrayElements(in_arr,JNI_FALSE);
        jdouble* u = env->GetDoubleArrayElements(u_arr, JNI_FALSE);
        jdouble* s = env->GetDoubleArrayElements(s_arr, JNI_FALSE);
        jdouble* v = env->GetDoubleArrayElements(v_arr, JNI_FALSE);

        int m = min( n,p);
        Map<MatrixXd> In(in, n, p );
        JacobiSVD<MatrixXd,HouseholderQRPreconditioner> svd(In, ComputeThinU | ComputeThinV);
        matrixToValues( n, m, &(svd.matrixU()), u );
        for( int i = 0; i < m; i++ ) {
            s[i] = svd.singularValues()(i);
        }
        matrixToValues( p, m, &(svd.matrixV()), v );
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_jeigen_eig( JNIEnv *env, jclass clazz, jint n, jdoubleArray in_arr, jdoubleArray values_real_arr, jdoubleArray values_imag_arr, jdoubleArray vectors_real_arr, jdoubleArray vectors_imag_arr ) {

        jdouble* in = env->GetDoubleArrayElements(in_arr,JNI_FALSE);
        jdouble* values_real = env->GetDoubleArrayElements(values_real_arr, JNI_FALSE);
        jdouble* values_imag = env->GetDoubleArrayElements(values_imag_arr, JNI_FALSE);
        jdouble* vectors_real = env->GetDoubleArrayElements(vectors_real_arr, JNI_FALSE);
        jdouble* vectors_imag = env->GetDoubleArrayElements(vectors_imag_arr, JNI_FALSE);

        Map<MatrixXd> In( in, n, n );
        EigenSolver<MatrixXd> eigenSolve( In );
        VectorXcd EigenValues = eigenSolve.eigenvalues();
        MatrixXcd EigenVectors = eigenSolve.eigenvectors();
        int i = 0;
        for ( int r = 0; r < n; r++ ) {
            values_real[i] = EigenValues(r).real();
            values_imag[i] = EigenValues(r).imag();
            i++;
        }
        i = 0;
        for( int c = 0; c < n; c++ ) {
            for ( int r = 0; r < n; r++ ) {
                vectors_real[i] = EigenVectors(r,c).real();
                vectors_imag[i] = EigenVectors(r,c).imag();
                i++;
            }
        }
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_jeigen_peig( JNIEnv *env, jclass clazz, jint n, jdoubleArray in_arr, jdoubleArray eigenValues_arr, jdoubleArray eigenVectors_arr ) {

        jdouble* in = env->GetDoubleArrayElements(in_arr,JNI_FALSE);
        jdouble* eigenValues = env->GetDoubleArrayElements(eigenValues_arr, JNI_FALSE);
        jdouble* eigenVectors = env->GetDoubleArrayElements(eigenVectors_arr, JNI_FALSE);


        Map<MatrixXd> In( in, n, n );
        EigenSolver<MatrixXd> eigenSolve( In );
        MatrixXd EigenValues = eigenSolve.pseudoEigenvalueMatrix();
        MatrixXd EigenVectors = eigenSolve.pseudoEigenvectors();
        matrixToValues( n, n, &(EigenValues), eigenValues );
        matrixToValues( n, n, &(EigenVectors), eigenVectors );
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_jeigen_1exp( JNIEnv *env, jclass clazz, jint n, jdoubleArray in_arr, jdoubleArray result_arr ) {

        jdouble* in = env->GetDoubleArrayElements(in_arr,JNI_FALSE);
        jdouble* result = env->GetDoubleArrayElements(result_arr, JNI_FALSE);

        Map<MatrixXd> In(in, n, n );
        Map<MatrixXd> Result(result,n,n);
        Result = In.exp();

//        print_log("result","expo:"+to_string(Result.cols()));
    }
    JNIEXPORT void JNICALL Java_com_example_datacollection_jeigen_jeigen_log(JNIEnv *env, jclass clazz, jint n, jdoubleArray in_arr, jdoubleArray result_arr ) {

        jdouble* in = env->GetDoubleArrayElements(in_arr,JNI_FALSE);
        jdouble* result = env->GetDoubleArrayElements(result_arr, JNI_FALSE);

        Map<MatrixXd> In(in, n, n );
        Map<MatrixXd> Result(result,n,n);
        Result = In.log();
    }
} // extern "C"

