/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author DCS
 */
import Jama.Matrix;
import com.xeiam.xchart.Chart;
import com.xeiam.xchart.QuickChart;
import com.xeiam.xchart.SwingWrapper;
import java.io.*;
import java.util.ArrayList;


/**
 * Implements multivariate linear regression. 
 * @author DCS
 */
public class MultivariateLR {

    protected static double alpha = 0.01;
    protected static int num_iters = 400;

    /**
     *FEATURENORMALIZE Normalizes the features in X 
     *   FEATURENORMALIZE(X) returns a normalized version of X where
     *   the mean value of each feature is 0 and the standard deviation
     *   is 1. This is often a good preprocessing step to do when
     *   working with learning algorithms.
     * working with learning algorithms.
     * @param X the matrix to be normalized
     * @return the object the contains the matrix values of X, mu and sigma
     */
     protected static FeatureNormalizationValues featureNormalize(Matrix X) {
        //Write equivalent Java code for the Octave code below.
        // You need to set these values correctly. 
        //Octave: X_norm = X;

        //Octave: mu = zeros(1, size(X, 2));

        //Octave: sigma = zeros(1, size(X, 2));


        // ====================== YOUR CODE HERE ======================
        // Instructions: First, for each feature dimension, compute the mean
        //               of the feature and subtract it from the dataset,
        //               storing the mean value in mu. Next, compute the 
        //               standard deviation of each feature and divide
        //               each feature by it's standard deviation, storing
        //               the standard deviation in sigma. 
        //
        //               Note that X is a matrix where each column is a 
        //               feature and each row is an example. You need 
        //               to perform the normalization separately for 
        //               each feature. 
        //
        // Hint: You might find the 'mean' and 'std' functions useful.
        //       
        Matrix X_norm = X;
        Matrix mu = new Matrix(1,X.getColumnDimension());
        Matrix sigma = new Matrix(1,X.getColumnDimension());

        //------------mean-------------
        for(int r=0;r<X_norm.getRowDimension();r++){
            for(int c=0;c<X_norm.getColumnDimension();c++){
                mu.set(0, c, mu.get(0,c)+X_norm.get(r, c));
                sigma.set(0,c,sigma.get(0,c)+Math.pow(X_norm.get(r, c),2));
            }
        }
        
        for(int c=0;c<X_norm.getColumnDimension();c++){
            double EX2 = sigma.get(0, c);
            double EX_2 = Math.pow(mu.get(0, c), 2)/X_norm.getRowDimension();
            
            sigma.set(0,c,Math.sqrt((EX2-EX_2)/(X_norm.getRowDimension()-1)));
            mu.set(0, c, mu.get(0, c)/(X_norm.getRowDimension()));
        }
        
        for(int r=0;r<X_norm.getRowDimension();r++){
            for(int c=0;c<X_norm.getColumnDimension();c++){
                X_norm.set(r,c,(X_norm.get(r, c)-mu.get(0, c))/sigma.get(0, c));
            }
        }
        return new FeatureNormalizationValues(X_norm, mu, sigma);
    }

    /**
     * GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
     * @param X
     * @param y
     * @param theta
     * @param alpha
     * @param num_iters
     * @return 
     */
     protected static GradientDescentValues gradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, int num_iters) {
        //Write equivalent Java code for the Octave code below.

        //Initialize some useful values.
        //Octave: m = length(y); % number of training examples

        //create a matrix that stores cost history
        //Octave: J_history = zeros(num_iters, 1);

        //Loop thru numIterations
        //Octave:for iter = 1:num_iters

        // ====================== YOUR CODE HERE ======================
        // Instructions: Perform a single gradient step on the parameter vector
        //               theta. 
        //
        // Hint: While debugging, it can be useful to print out the values
        //       of the cost function (computeCostMulti) and gradient here.
        //
         int m = X.getRowDimension();
         Matrix J_history = new Matrix(num_iters, 1);
         
         for(int iter=0;iter<num_iters;iter++){    
             J_history.set(iter, 0, computeCostMulti(X, y, theta));
             theta = theta.minus(X.transpose().times(X.times(theta).minus(y)).times(alpha/m));
             
             
             //declare convergence if current cost is less than 0.001 by the previous cost
             if(iter>0){
                 double diff = J_history.get(iter-1, 0)-J_history.get(iter, 0);
                 if(diff<0.001){
                     System.out.println("ended at iteration no: " + (iter+1));
                     break;
                 }
             }
         }
        // Save the cost J in every iteration    
        //Octave: J_history(iter) = computeCostMulti(X, y, theta);
        return new GradientDescentValues(theta, J_history);
    }

    /**
     *COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
     *   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
     *   parameter for linear regression to fit the data points in X and y
     * @param X
     * @param y
     * @param theta
     * @return 
     */
     protected static double computeCostMulti(Matrix X, Matrix y, Matrix theta) {
        //Write equivalent Java code for the Octave code below.
        // Initialize some useful values
        //Octave: m = length(y); % number of training examples

        // You need to return the following variables correctly 
        //Octave: J = 0;

        // ====================== YOUR CODE HERE ======================
        // Instructions: Compute the cost of a particular choice of theta
        //                You should set J to the cost.
         int m = y.getRowDimension();
         Matrix sum = X.times(theta).minus(y);
         return sum.transpose().times(sum).times(1.d/(2.d*m)).get(0, 0);
     }
     

    /**
    NORMALEQN Computes the closed-form solution to linear regression 
    NORMALEQN(X,y) computes the closed-form solution to linear 
    regression using the normal equations.
     * @param X
     * @param y
     * @return 
     */
    protected static Matrix normalEqn(Matrix X, Matrix y) {
        //Write equivalent Java code for the Octave code below.

        //Octave: theta = zeros(size(X, 2), 1);

        // ====================== YOUR CODE HERE ======================
        //        Instructions: Complete the code to compute the closed form solution
        //               to linear regression and put the result in theta.
        //Matrix theta = new Matrix(X.getColumnDimension(),1);
        return (X.transpose().times(X)).inverse().times(X.transpose().times(y));        
    }
}

class GradientDescentValues {

    Matrix theta;
    Matrix costHistory;
    
    public GradientDescentValues(Matrix _theta, Matrix _costHistory){
        theta = _theta;
        costHistory = _costHistory;
    }
    
    public Matrix getTheta() {
        return theta;
    }

    public void setTheta(Matrix theta) {
        this.theta = theta;
    }

    public Matrix getCostHistory() {
        return costHistory;
    }

    public void setCostHistory(Matrix costHistory) {
        this.costHistory = costHistory;
    }
}

class FeatureNormalizationValues {

    Matrix X;
    Matrix mu;
    Matrix sigma;
    
    public FeatureNormalizationValues(Matrix _X, Matrix _mu, Matrix _sigma){
        X = _X;
        mu = _mu;
        sigma = _sigma;
    }

    public Matrix getX() {
        return X;
    }

    public void setX(Matrix X) {
        this.X = X;
    }

    public Matrix getMu() {
        return mu;
    }

    public void setMu(Matrix mu) {
        this.mu = mu;
    }

    public Matrix getSigma() {
        return sigma;
    }

    public void setSigma(Matrix sigma) {
        this.sigma = sigma;
    }
}

class TestMLR extends Thread{
    
    double[] J;
    
    public TestMLR(double[] _J){
        J = _J;
    }
    
    @Override
    public void run() {
        Chart c = QuickChart.getChart("Cost x iterations plot","iterations","J(theta)",null,null,J);
        new SwingWrapper(c).displayChart();
    }

    public static void main(String[] args) throws FileNotFoundException, IOException {
        //Write the corresponding Java code for the Octave code below between /**...*/

        long start = System.currentTimeMillis();

        System.out.println("Loading data...");
        //code for loading data
        /**
        data = load('ex1data2.txt');
        X = data(:, 1:2);
        y = data(:, 3);
        m = length(y);
        %Print out some data points
        fprintf('First 10 examples from the dataset: \n');
        fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
         */
        //JAVA CODE HERE
        //==============
        String file = "ex1data2.txt";
        Matrix X = data(file, 1, 2);
        Matrix y = data(file, 3);
        int m = y.getRowDimension();

        System.out.println("First 10 examples from the dataset: n");
        fprintfMatrix(X, y, 10);
        
        //==============

        System.out.println("Normalizing features..");
        //code for normalizing data
        /**
        [X mu sigma] = featureNormalize(X);
        
        % Add intercept term to X
        X = [ones(m, 1) X];
         */
        //JAVA CODE HERE
        //==============
        FeatureNormalizationValues f = MultivariateLR.featureNormalize(X);
        X = insertX0(f.X);
        //==============
        System.out.println("Running gradient descent...");
        //code for performing gradientDescent
        /**
        % Choose some alpha value
        alpha = 0.01;
        num_iters = 400;
        
        % Init Theta and Run Gradient Descent 
        theta = zeros(3, 1);
        [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
        
        % Plot the convergence graph
        figure;
        plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
        xlabel('Number of iterations');
        ylabel('Cost J');
        
        % Display gradient descent's result
        fprintf('Theta computed from gradient descent: \n');
        fprintf(' %f \n', theta);
         */
        //JAVA CODE HERE
        //==============
        //------alpha and num_iters already in class MultivariateLR-----
        Matrix theta = new Matrix(X.getColumnDimension(),1);
        Matrix J_history = new Matrix(MultivariateLR.num_iters,1);
        
        GradientDescentValues g = MultivariateLR.gradientDescent(X, y, theta, MultivariateLR.alpha, MultivariateLR.num_iters);
        theta = g.getTheta();
        J_history = g.getCostHistory();
        
        //**********PLOTTING****************
        //-----> create new thread for faster calculation while plotting :D
        //test mlr constructor gets the dataset for y: J-theta
        new Thread(new TestMLR(J_history.getRowPackedCopy())).start();
        //==============
        System.out.println("Estimating price...");

        /*
        % Estimate the price of a 1650 sq-ft, 3 br house
        % ====================== YOUR CODE HERE ======================
        % Recall that the first column of X is all-ones. Thus, it does
        % not need to be normalized.
        price = 0; % You should change this
        
        
        % ============================================================
        
        fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
        '(using gradient descent):\n $%f\n'], price);
         */
        //JAVA CODE HERE
        //==============
        double price = 0.0;
        
        Matrix val1 = new Matrix(X.getColumnDimension(),1);
        val1.set(0, 0, 1);
        val1.set(1,0,((1650-f.mu.get(0, 0))/f.sigma.get(0, 0)));
        val1.set(2,0,((3-f.mu.get(0, 1))/f.sigma.get(0, 1)));
        price = theta.transpose().times(val1).get(0, 0);
        System.out.println();
        
        System.out.println("Theta computed from gradient descent: ");
        theta.print(3, 4);
        System.out.println();
        
        System.out.println("Predicted price of a 1650 sq-ft, 3br "
                + "house\n(using gradient descent):\n"
                + price +"\n\n" + "**********************\n");
        
        //==============

        System.out.println("Solving with normal equations");
        /**
        %% Load Data
        data = csvread('ex1data2.txt');
        X = data(:, 1:2);
        y = data(:, 3);
        m = length(y);
        
        % Add intercept term to X
        X = [ones(m, 1) X];
        
        % Calculate the parameters from the normal equation
        theta = normalEqn(X, y);
        
        % Display normal equation's result
        fprintf('Theta computed from the normal equations: \n');
        fprintf(' %f \n', theta);
        fprintf('\n');
        
        
        % Estimate the price of a 1650 sq-ft, 3 br house
         */
        //====================== YOUR CODE HERE ======================
        //price = 0; % You should change this

        
        X = data(file, 1, 2);
        X = insertX0(X);
        
        theta = MultivariateLR.normalEqn(X, y);
        System.out.println("Theta computed from the normal equations: ");
        theta.print(3, 4);
        System.out.println();
        
        /*% ============================================================
        
        fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
        '(using normal equations):\n $%f\n'], price);
         */
        //JAVA CODE HERE
        //==============
        System.out.println("Predicted price of a 1650 sq-ft, 3 br house ");
        System.out.println("(using normal equations):");
        Matrix val = new Matrix(X.getColumnDimension(),1);
        val.set(0,0,1);
        val.set(1,0,1650);
        val.set(2,0,3);
        
        price = theta.transpose().times(val).get(0, 0);
        
        System.out.println(price);
        
        
        for (int i = 0; i < 1000000000; i++) {
            double a = Math.sqrt((i + 5.9) * (i * i));
        }

        //==============
        long end = System.currentTimeMillis();

        long dif = end - start;
        if (dif > 1000) {
            dif = (end - start) / 1000;
            System.out.println("Speed:" + dif + " seconds");
        } else {
            System.out.println("Speed:" + dif + " milliseconds");
        }


    }

    //-----------------custom made functions----------------------
    /**
     * data returns vector data set with specified column
     * file: data file
     * col: starting index
     * @param filename
     * @param start
     * @return Matrix
     */
    private static Matrix data(String filename, int col) throws FileNotFoundException, IOException {
        BufferedReader bf = new BufferedReader(new FileReader(filename));
        col--;
        String line = bf.readLine();
        String colval = "";
        ArrayList<Double> training_examples = new ArrayList<Double>();

        while (line != null) {
            colval = line.split(",")[col];
            training_examples.add(Double.valueOf(colval));
            line = bf.readLine();
        }
        
        int r = training_examples.size();
        Matrix matrix = new Matrix(r, 1);
        
        for (int i = 0; i < training_examples.size(); i++) 
            matrix.set(i, 0, training_examples.get(i));

        return matrix;
    }

    /**
     * data returns data set using matrix with specified column
     * file: data file
     * start: starting index
     * end: end index
     * @param filename
     * @param start
     * @return Matrix
     */
    private static Matrix data(String filename, int start, int end) throws FileNotFoundException, IOException {
        start--; //index's sake
        end--;
        
        BufferedReader bf = new BufferedReader(new FileReader(filename));
        String line = bf.readLine();
        String[] field = line.split(",");
        ArrayList<double[]> training_examples = new ArrayList<double[]>();
        
        while (line != null) {
            field = line.split(",");
            double[] data = new double[(end-start)+1];
            
            for (int i = start; i <= end-start; i++) {
                data[i] = Double.parseDouble(field[i]);
            }
            
            training_examples.add(data);
            line = bf.readLine();
        }

        int c = training_examples.get(0).length;
        int r = training_examples.size();

        Matrix matrix = new Matrix(r, c);
        for (int rr = 0; rr < r; rr++) {
            for (int cc = 0; cc < c; cc++) {
                matrix.set(rr, cc, training_examples.get(rr)[cc]);
            }
        }

        return matrix;
    }
    
    private static Matrix insertX0(Matrix F){
        Matrix _X = new Matrix(F.getRowDimension(),F.getColumnDimension()+1);
        for(int r=0;r<F.getRowDimension();r++){
            for(int c=0;c<=F.getColumnDimension();c++){
                if(c==0)
                    _X.set(r, c, 1);
                else
                    _X.set(r, c, F.get(r, c-1));
            }
        }
        return _X;
    }
    
    private static void fprintfMatrix(Matrix X, Matrix Y, int rows){
        for(int r=0;r<rows;r++){
            for(int c=0;c<X.getColumnDimension();c++){
                System.out.print(X.get(r, c) + " ");
            }
            System.out.println(" " + Y.get(r, 0));
        }
    }
    
    public static void fprintfMatrix(Matrix X, int rows){
        for(int r=0;r<rows;r++){
            for(int c=0;c<X.getColumnDimension();c++){
                System.out.print(X.get(r, c) + " ");
            }
            System.out.println();
        }
    }
}