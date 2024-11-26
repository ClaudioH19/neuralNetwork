package neuralnetwork;
import java.util.random.*;
public class Neuron {

    double [] weights_behind;
    double bias;

    double z;
    double prediction;
    double delta_error;


    public Neuron(int tam_capa) {

        this.weights_behind = new double[tam_capa];
        for (int i = 0; i < tam_capa; i++) {
            weights_behind[i] = Math.random()*0.5-0.5;

            //initializeWeightsXavier(tam_capa);
        }
        this.bias = Math.random()-1;

        this.z = 0;
        this.prediction = 0;
    }



    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoid_derivated(double x){
        return x*(1-x);
    }

    public double error_square(double target, double actual){
        return (0.5*Math.pow(target-actual,2));
    }

    public double error_square_derivated(double actual,double target){
        return actual-target;
    }

    public double calculate_z(double[] inputs){
        z=0;
        for (int i=0;i<inputs.length;i++){
            z+=inputs[i]*weights_behind[i];
        }
        z+=bias;
        return z;
    }

    public double calculate_prediction(){
        prediction=sigmoid(z);
        return prediction;
    }

    public void initializeWeightsXavier(int inputstam) {
        for (int i = 0; i < weights_behind.length; i++) {
            weights_behind[i] =  Math.random()* Math.sqrt(1.0 / inputstam);
        }
    }

}
