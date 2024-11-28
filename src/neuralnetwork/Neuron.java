package neuralnetwork;
import java.util.random.*;
public class Neuron {
    //ESTA CLASE TIENE LAS HERRAMIENTAS PARA MANEJAR NEURONAS
    double [] weights_behind; //una neurona guarda los pesos de las conexiones con la capa de su izquierda
    double bias;

    double z;
    double prediction;
    double delta_error;




    public Neuron(int tam_capa) {
        this.weights_behind = new double[tam_capa];
        for (int i = 0; i < tam_capa; i++) {
            //elegimos un tipo de inicializacion

            //weights_behind[i] = Math.random()*0.5-0.5;

            //initializeWeightsXavier(tam_capa);
            initializeWeightsHe(tam_capa);
        }

        //this.bias = Math.random()-1;
        this.bias=0;
        this.z = 0;
        this.prediction = 0;
    }


    //FUNCIONES DE ACTIVACION DISPONIBLES
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double relu(double x){
        if(x>0) return x;
        else return 0;
    }

    public double[] softmax(double[] z) {
        double sum = 0.0;
        double[] result = new double[z.length];

        // Cálculo de exp(z) para cada elemento
        for (int i = 0; i < z.length; i++) {
            result[i] = Math.exp(z[i]); // Elevar cada valor a e^z
            sum += result[i]; // Sumar las exponentes
        }

        // Normalizar dividiendo por la suma total
        for (int i = 0; i < z.length; i++) {
            result[i] /= sum; // Convertir a probabilidades
        }

        return result;
    }

    //Se usa como controlador para elegir una funcion de activacion
    public double calculate_prediction(String method){
        switch (method){
            case "sigmoid":
                prediction=sigmoid(z);
                break;
            case "relu":
                prediction=relu(z);
                break;
            default:
                prediction = z;
                break;
        }
        return prediction;
    }



    //DERIVADAS DE LAS FUNCIONES
    public double sigmoid_derivated(double x){
        return x*(1-x);
    }

    public double relu_derivated(double x){
        if(x>0) return 1;
        else return 0;
    }

    //FUNCIONES DE ERROR DISPONIBLES
    public double error_square(double target, double actual){
        return (0.5*Math.pow(target-actual,2));
    }

    public double crossEntropy(double target, double prediction) {
        return -target * Math.log(prediction);
    }

    //DERIVADAS DE FUNCIONES DE ERROR

    public double error_square_derivated(double actual,double target){
        return actual-target;
    }

    public double crossEntropyDerivative(double target, double prediction) {
        return -target / prediction;
    }



    //CALCULAR Z con las entradas de una neurona wx+b
    public double calculate_z(double[] inputs){
        z=0;
        for (int i=0;i<inputs.length;i++){
            z+=inputs[i]*weights_behind[i];
        }
        z+=bias;
        return z;
    }



    //TIPOS DE INICIALIZACIONES
    public void initializeWeightsXavier(int inputstam) {
        for (int i = 0; i < weights_behind.length; i++) {
            weights_behind[i] =  Math.random()* Math.sqrt(1.0 / inputstam);
        }
    }

    public void initializeWeightsHe(int tam_capa) {
        for (int i = 0; i < weights_behind.length; i++) {
            weights_behind[i] = (Math.random()*2-1 + Math.random()*2-1) * Math.sqrt(2.0/tam_capa);
        }
    }

}
