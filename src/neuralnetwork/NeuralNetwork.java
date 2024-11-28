/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

/**
 *
 * @author claud
 */
public class NeuralNetwork {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here

        double[][] inputs = new double[4][2];
        double[][] outputs = new double[4][2];

        String wine = "red"; //<--- controla que vinos predecir white-red

        //para datos del vino rojo
        if(wine.equals("red")){
            int limit=1599; //el limite de lectura para los datos: el idx del último dato
            Datos datos = new Datos(limit,wine);

            //11 neuronas con 1 capa de input y 2 capas: 1 hidden - 1 outputs
            Network net = new Network(11, datos.inputs, datos.outputs,2);

            net.iterar(10000,1280); //desencadena el backpropagation y entrena hasta el idx 1280

            //usamos un 80% de datos para entrenar y el 20% para testear
            System.out.println("Testing");
            net.testing(1280); // usa los datos restantes a partir de 1280 para evaluar rendimiento
        }

        //para datos del vino blanco
        else{
            int limit=4899;
            Datos datos = new Datos(limit,wine);
            Network net = new Network(11, datos.inputs, datos.outputs,2);
            net.iterar(10000,3919);
            System.out.println("Testing");
            net.testing(3919);
        }

    }
    
}
