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


        //xor
        inputs[0][0]=0;
        inputs[0][1]=0;

        inputs[1][0]=1;
        inputs[1][1]=0;

        inputs[2][0]=0;
        inputs[2][1]=1;

        inputs[3][0]=1;
        inputs[3][1]=1;

        outputs[0][0]=0;
        outputs[0][1]=0;

        outputs[1][0]=1;
        outputs[1][1]=0;

        outputs[2][0]=0;
        outputs[2][1]=1;

        outputs[3][0]=0;
        outputs[3][1]=0;


        double learning_rate = 0.0005;
        learning_rate = 0.0005;
        //Network net = new Network(2, inputs, outputs,learning_rate,2);


        int limit=1599;
        Datos datos = new Datos(limit);
        Network net = new Network(11, datos.inputs, datos.outputs,learning_rate,2);
        net.iterar(20000,1280);
        System.out.println("Testing");
        net.testing(1280);
    }
    
}
