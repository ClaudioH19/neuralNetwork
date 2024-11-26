package neuralnetwork;
import java.util.ArrayList;
import java.util.Arrays;


public class Network {

    double learning_rate;
    double[][] inputs;
    double[][] outputs_expected;
    double[] output_predictions;

    ArrayList<ArrayList> layers;

    int tam_capa;

    public Network(int tam_capa, double[][] inputs, double[][] outputs,double learning_rate, int tam_layers) {
        this.learning_rate = learning_rate;
        this.output_predictions =new double[tam_capa];
        this.tam_capa=tam_capa;
        this.inputs = inputs;
        this.outputs_expected=outputs;
        this.layers = new ArrayList<>();


        //crear capas
        for (int i=0;i<tam_layers;i++){
            ArrayList<Neuron> l=new ArrayList<>();
            for (int j=0;j<tam_capa;j++){
                l.add(new Neuron(tam_capa));
            }
            layers.add(l);
        }

    }

    //Descompondremos la matriz de inputs para ir revisando fila por fila
    public void forward(double[] inputs) {

        ///calcular z y predicciones para todas las neuronas
        double[] predictions = new double[tam_capa];

        ///por cada capa
        for (int i=0;i<layers.size();i++){
            ///sacamos sus neuronas
            ArrayList<Neuron> l=layers.get(i);

            int iter=0;
            ///calculamos los z y predicciones
            for(Neuron n : l){
                //que entradas usar
                if(i==0)
                    n.calculate_z(inputs);
                else
                    n.calculate_z(predictions);

                ///donde guardar si es la ultima capa o no
                if(i==layers.size()-1)
                    output_predictions[iter]=n.calculate_prediction();
                else
                    predictions[iter]=n.calculate_prediction();
            }
        }
    }

    public double backward(double[] outputs_expected) {

        //recorrer cada capa
        for (int i=layers.size()-1;i>=0;i--){

            ///sacar capas actual: l y capa de la derecha: l_d
            ArrayList<Neuron> l=layers.get(i);
            ArrayList<Neuron> l_d=new ArrayList<>();
            if(i<layers.size()-1)
                l_d=layers.get(i+1);


            int iter=0;

            //Recorrer cada neurona de la capa l
            for(Neuron n : l) {

                ///recoger gradientes para la capa salida
                if (i == layers.size() - 1) {
                                    ///esta es la derivada del error cuadrado
                    n.delta_error = n.error_square_derivated (n.prediction,outputs_expected[iter]) * n.sigmoid_derivated(n.prediction);
                    iter++;
                }

                ///recoger gradientes para las capas ocultas
                else{
                    double sum=0;
                    for(int j=0;j<tam_capa;j++) {
                        //vemos la capa de la derecha pues esta contiene los pesos
                        sum += l_d.get(j).delta_error * l_d.get(j).weights_behind[iter];
                    }
                    n.delta_error=sum* n.sigmoid_derivated(n.prediction);
                    iter++;
                }
            }
        }

        return 0;
    }




    public void learning(double[] inputs){
    ///recorrer las capas
        for (int k=layers.size()-1;k>=0;k--){
            ///sacar capa actual y la de la izquierda
            ArrayList<Neuron> l=layers.get(k);
            ArrayList<Neuron> l_i=new ArrayList<>();
            if(k>0)
                l_i=layers.get(k-1);

            for(Neuron n : l){
                for (int i = 0; i < tam_capa; i++) {
                    ///actualizar pesos y bias

                    ///para las capas ocultas y la de salida
                    if(k>0)
                        n.weights_behind[i] -= learning_rate * n.delta_error * l_i.get(i).prediction;
                    ///para la capa conectada a las entradas
                    else
                        n.weights_behind[i]-=learning_rate * n.delta_error * inputs[i];

                    n.bias -= learning_rate * n.delta_error;
                    }
                }
            }
    }


    public void iterar(int limit, int limit_entrenamiento){

        for (int age=0;age<limit;age++) {

            for (int fila=0;fila<inputs.length && fila<limit_entrenamiento;fila++) {
                forward(inputs[fila]);
                backward(outputs_expected[fila]);
                learning(inputs[fila]);
            }
            //System.out.println("NEW ERA");
        }

    }

    public void testing(int ini_testeo){
        //testear
        //en la capa salida deberían los h(z) estar en este orden 00, 10 , 01, 00 por cada fila para xor

        ArrayList<Neuron> output=layers.get(layers.size()-1);
        for (int fila=ini_testeo;fila<inputs.length;fila++) {
            forward(inputs[fila]);
            for(Neuron n : output){
                System.out.print(" "+n.prediction);
            }
            System.out.println();
        }


        int hits=0;
        int misses=0;
        System.out.println("aproximado: ");
        for (int fila=ini_testeo;fila<inputs.length;fila++) {

            forward(inputs[fila]);
            double[] aproximados = new double[inputs[0].length];
            double[] resultados = new double[inputs[0].length];
            int iter=0;
            double max=-1000;
            //buscar el valor max
            for(Neuron n : output){
                if(n.prediction>max)
                    max=n.prediction;
            }
            for(Neuron n : output){
                //double num=Math.round(n.prediction);
                double num=0;
                //if (n.prediction>0.45) num=1;

                if(n.prediction==max) num=1;
                aproximados[iter]=num;
                resultados[iter]=outputs_expected[fila][iter];
                System.out.print(" "+num);
                iter++;
            }
            //System.out.println();
            //System.out.println(Arrays.toString(aproximados));
            //System.out.println(Arrays.toString(resultados));

            if(Arrays.equals(aproximados,resultados)){
                hits++;
            }
            else{
                misses++;
            }
            System.out.println();
        }
        double tax= (double) hits /(hits+misses);
        System.out.println("HITS: "+hits+" MISSES: "+misses+" TASA IGUALDAD: "+ tax);
    }
}
