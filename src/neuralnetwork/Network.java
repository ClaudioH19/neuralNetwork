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
            for (Neuron n : l) {
                if (i == 0) { // Primera capa: usar las entradas
                    n.calculate_z(inputs);
                } else { // Capas ocultas y de salida: usar las predicciones de la capa previa
                    n.calculate_z(predictions);
                }

                // Guardar las predicciones de cada neurona
                if (i == layers.size() - 1) { // Última capa: no aplicar activación aquí
                    output_predictions[iter] = n.calculate_prediction("none");
                } else { // Capas ocultas: aplicar ReLU
                    predictions[iter] = n.calculate_prediction("relu");
                }
                iter++;
            }
        }
        // Aplicar Softmax a las salidas de la última capa
        Neuron aux=new Neuron(tam_capa);
        output_predictions = aux.softmax(output_predictions);
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
                    ///esta es la derivada del error cuadrado con sigmoid
                    //n.delta_error = n.error_square_derivated (n.prediction,outputs_expected[iter]) * n.sigmoid_derivated(n.prediction);

                    //esta es la derivada de cross entropy con softmax
                    n.delta_error =n.prediction - outputs_expected[iter];
                }

                ///recoger gradientes para las capas ocultas
                else{
                    double sum=0;
                    for(int j=0;j<tam_capa;j++) {
                        //vemos la capa de la derecha pues esta contiene los pesos
                        sum += l_d.get(j).delta_error * l_d.get(j).weights_behind[iter];
                    }

                    ///para sigmoid
                    //n.delta_error=sum* n.sigmoid_derivated(n.prediction);
                    ///para relu
                    n.delta_error=sum* n.relu_derivated(n.prediction);
                }
                iter++;
                //System.out.print(n.delta_error );
            }
            //System.out.println();
        }

        return 0;
    }




    public void learning(double[] inputs){
        double lambda=0.00001;
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
                        n.weights_behind[i] -= learning_rate * (n.delta_error * l_i.get(i).prediction + lambda* n.weights_behind[i]);
                    ///para la capa conectada a las entradas
                    else
                        n.weights_behind[i]-=learning_rate * (n.delta_error * inputs[i]+ lambda* n.weights_behind[i]);

                    n.bias -= learning_rate * (n.delta_error +lambda* n.bias);
                    }
                }
            }
    }

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


    public void iterar(int limit, int limit_entrenamiento){

        double initialLearningRate = learning_rate;
        double decayRate = 0.9999;

        for (int age=0;age<limit;age++) {
            learning_rate = initialLearningRate * Math.pow(decayRate, age);


            for (int fila=0;fila<inputs.length && fila<limit_entrenamiento;fila++) {
                forward(inputs[fila]);
                backward(outputs_expected[fila]);
                learning(inputs[fila]);
            }

            double loss = calculateLoss();
            //System.out.println("Época: " + age + " Pérdida Promedio: " + loss);
        }

    }


    public void testing(int ini_testeo) {
        int hits = 0;
        int misses = 0;

        System.out.println("Resultados de testeo: ");
        for (int fila = ini_testeo; fila < inputs.length; fila++) {
            forward(inputs[fila]); // Realiza el paso hacia adelante

            double[] aproximados = new double[outputs_expected[0].length];
            double[] resultados = outputs_expected[fila];

            // identificar max proba
            int predictedClass = -1;
            double maxProbability = -1.0;
            for (int i = 0; i < output_predictions.length; i++) {
                if (output_predictions[i] > maxProbability) {
                    maxProbability = output_predictions[i];
                    predictedClass = i;
                }
            }

            // dejar en formato arreglo
            for (int i = 0; i < aproximados.length; i++) {
                aproximados[i] = (i == predictedClass) ? 1 : 0;
            }

            if (Arrays.equals(aproximados, resultados)) {
                hits++;
            } else {
                misses++;
            }

            System.out.println("Predicción: " + Arrays.toString(aproximados) +
                    " Esperado: " + Arrays.toString(resultados));
        }

        // Calcular y mostrar la tasa de aciertos
        double accuracy = (double) hits / (hits + misses);
        System.out.println("HITS: " + hits + " MISSES: " + misses + " TASA DE ACIERTOS: " + accuracy);
    }




}
