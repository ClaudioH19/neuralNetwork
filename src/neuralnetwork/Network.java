package neuralnetwork;
import java.util.ArrayList;
import java.util.Arrays;


public class Network {
    //PARAMETROS
    //double learning_rate = 0.0005;
    //double lambda=0.00001;
    //double decayRate = 0.9999;
    //testing
    //double learning_rate = 0.0005;
    //double lambda=0.000007;
    //double decayRate = 0.99999;

    //testing
    double learning_rate = 0.005;
    double lambda=0.0000007;
    double decayRate = 0.99999;

    double[][] inputs;
    double[][] outputs_expected;
    double[] output_predictions;

    ArrayList<ArrayList> layers;
    int tam_capa;

    public Network(int tam_capa, double[][] inputs, double[][] outputs, int tam_layers) {
        this.output_predictions =new double[tam_capa];
        this.tam_capa=tam_capa;
        this.inputs = inputs;
        this.outputs_expected=outputs;
        this.layers = new ArrayList<>();

        //crear capas de neuronas
        for (int i=0;i<tam_layers;i++){
            ArrayList<Neuron> l=new ArrayList<>();
            for (int j=0;j<tam_capa;j++){
                l.add(new Neuron(tam_capa));
            }
            layers.add(l);
        }

    }

    //recorre cada capa de izq a der y cada neurona de una capa para calcular su z y su h(z) o funcion de activacion
    public void forward(double[] inputs) {

        double[] predictions = new double[tam_capa];

        //por cada capa
        for (int i=0;i<layers.size();i++){
            //tomamos sus neuronas
            ArrayList<Neuron> l=layers.get(i);

            int iter=0;
            //iteramos por cada neurona de una capa y calculamos su z
            for (Neuron n : l) {
                //si es primera capa usa los inputs para calcular
                if (i == 0) {
                    n.calculate_z(inputs);
                }
                //si es capa hidden, usa las salidas (h(z)) de las capas previas
                else {
                    n.calculate_z(predictions);
                }

                // si es ultima capa, no aplicamos una funcion (se aplica softmax mas adelante)
                if (i == layers.size() - 1) {
                    output_predictions[iter] = n.calculate_prediction("none"); //guardamos las salidas
                }
                // si es capa hidden, aplicamos ReLu
                else {
                    predictions[iter] = n.calculate_prediction("relu");
                }

                iter++;
            }
        }

        // terminado el proceso aplicamos softmax a la capa de salida
        Neuron aux=new Neuron(tam_capa);
        output_predictions = aux.softmax(output_predictions);
    }


    //recorremos de der a izq cada capa y cada neurona de una capa, para calcular error y gradientes
    public double backward(double[] outputs_expected) {

        //recorrer cada capa de der a izq
        for (int i=layers.size()-1;i>=0;i--){
            ArrayList<Neuron> l=layers.get(i); //llamamos l a la capa en la que estamos
            ArrayList<Neuron> l_d=new ArrayList<>();//lamamos l_d a la capa a su derecha

            if(i<layers.size()-1)
                l_d=layers.get(i+1);


            int iter=0;
            //Recorrer cada neurona de la capa l
            for(Neuron n : l) {

                //Calcular gradientes para la capa salida
                if (i == layers.size() - 1) {
                    //esta es la derivada de cross entropy con softmax
                    n.delta_error = n.prediction - outputs_expected[iter];

                    //[en desuso, por peor rendimiento]
                    //esta es la derivada del error cuadrado con sigmoid
                    //n.delta_error = n.error_square_derivated (n.prediction,outputs_expected[iter]) * n.sigmoid_derivated(n.prediction);

                }

                //Calcular gradientes para las capas ocultas
                else{
                    double sum=0;
                    for(int j=0;j<tam_capa;j++) {
                        //vemos la capa de la derecha pues esta contiene los pesos y gradiente que necesitamos para el calculo
                        sum += l_d.get(j).delta_error * l_d.get(j).weights_behind[iter];
                    }

                    //Funcion de error derivada para relu
                    n.delta_error=sum* n.relu_derivated(n.prediction);

                    //Funcion de error derivada para sigmoid [en desuso]
                    //n.delta_error=sum* n.sigmoid_derivated(n.prediction);
                }
                iter++;
                //System.out.println(n.delta_error );
            }
        }
        return 0;
    }



    //Una vez calculado las predicciones de las neuronas y sus gradientes, usamos esta funcion para ajustar pesos
    public void learning(double[] inputs){

        //recorremos cada capa de der a izq
        for (int k=layers.size()-1;k>=0;k--){

            ArrayList<Neuron> l=layers.get(k); //llamamos l a la capa que estamos mirando
            ArrayList<Neuron> l_i=new ArrayList<>(); //lamamos l_i a la capa de la izquierda
            if(k>0)
                l_i=layers.get(k-1);

            //por cada neurona de la capa, ajustamos sus pesos y bias
            for(Neuron n : l){
                for (int i = 0; i < tam_capa; i++) {

                    //ajuste para las capas ocultas y la de salida + parametro lambda
                    if(k>0)
                        n.weights_behind[i] -= learning_rate * (n.delta_error * l_i.get(i).prediction + lambda* n.weights_behind[i]);
                    //ajuste para la capa conectada a las entradas + parametro lambda
                    else
                        n.weights_behind[i]-=learning_rate * (n.delta_error * inputs[i]+ lambda* n.weights_behind[i]);

                    //ajuste bias
                    n.bias -= learning_rate * (n.delta_error + lambda* n.bias);
                    }
                }
            }
    }

    //funcion para controlar la perdida
    public double calculateLoss() {
        double totalLoss = 0.0;
        for (int fila = 0; fila < inputs.length; fila++) {
            forward(inputs[fila]);
            for (int i = 0; i < outputs_expected[fila].length; i++) {
                totalLoss += -outputs_expected[fila][i] * Math.log(output_predictions[i]);
            }
        }
        return totalLoss / inputs.length;
    }


    //aqui se realiza el backpropagation
    public void iterar(int limit, int limit_entrenamiento){

        double initialLearningRate = learning_rate;
        for (int age=0;age<limit;age++) {
            //variamos el learning_rate por epoca
            learning_rate = initialLearningRate * Math.pow(decayRate, age);

            //por cada entrada entrenamos la red
            for (int fila=0;fila<inputs.length && fila<limit_entrenamiento;fila++) {
                forward(inputs[fila]);
                backward(outputs_expected[fila]);
                learning(inputs[fila]);
            }

            double loss = calculateLoss();
            //System.out.println("Época: " + age + " Pérdida Promedio: " + loss);
        }

    }


    //testeamos en los datos no vistos
    public void testing(int ini_testeo) {
        int hits = 0;
        int misses = 0;

        System.out.println("Resultados de testeo: ");
        for (int fila = ini_testeo; fila < inputs.length; fila++) {

            //hacemos un forward para ver las salidas de la red
            forward(inputs[fila]);

            double[] aproximados = new double[outputs_expected[0].length];
            double[] resultados = outputs_expected[fila];

            // identificar max proba en base a softmax
            int predictedClass = -1;
            double maxProbability = -1.0;
            for (int i = 0; i < output_predictions.length; i++) {
                if (output_predictions[i] > maxProbability) {
                    maxProbability = output_predictions[i];
                    predictedClass = i;
                }
            }

            // dejar en formato arreglo para comparar con los resultados esperados
            for (int i = 0; i < aproximados.length; i++) {
                aproximados[i] = (i == predictedClass) ? 1 : 0;
            }

            if (Arrays.equals(aproximados, resultados)) {
                hits++;
            } else {
                misses++;
            }

            System.out.println("Predicción: " + Arrays.toString(aproximados) + " Esperado: " + Arrays.toString(resultados));
        }

        // Calcular y mostrar la tasa de aciertos
        double accuracy = (double) hits / (hits + misses);
        System.out.println("HITS: " + hits + " MISSES: " + misses + " TASA DE ACIERTOS: " + accuracy);
    }




}
