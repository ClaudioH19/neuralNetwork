package neuralnetwork;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Datos {
    double[][] inputs;
    double[][] outputs;

    public Datos(int limit) {

        String csvFile = "C:\\Users\\claud\\Desktop\\winequality-white.csv";
        List<double[]> data = new ArrayList<>();
        List<double[]> out = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line = br.readLine(); //leer encabezado

            int count=0;
            while ((line = br.readLine()) != null && count < limit) {
                String[] stringValues = line.split(";");
                double[] doubleValues = new double[stringValues.length-1];

                double[] o = {0,0,0,0,0,0,0,0,0,0,0};
                //hacer la conversión del quality en formato binario
                int idx=Integer.parseInt(stringValues[stringValues.length-1]);
                o[idx]=1;
                out.add(o);

                //no contar las salidas
                for (int i = 0; i < stringValues.length-1; i++) {
                    doubleValues[i] = Double.parseDouble(stringValues[i].trim());
                }
                data.add(doubleValues);
                count++;
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }



        // Convertimos las listas a una matriz double[][]
        inputs = data.toArray(new double[0][]);
        outputs = out.toArray(new double[0][]);

        //printData();
        normalize();
        printData();

    }




    public void normalize(){
        //normalización
        double min=100000;
        double max=-100000;
        for (int j=0;j<inputs[0].length;j++) {
            //encontrar max y min de una col
            for (int i=0;i<inputs.length;i++) {
                if (inputs[i][j]<min) {
                    min=inputs[i][j];
                }
                if (inputs[i][j]>max) {
                    max=inputs[i][j];
                }
            }
            //actualizar
            for (int i=0;i<inputs.length;i++) {
                inputs[i][j]=(inputs[i][j]-min)/(max-min);
            }
        }
    }


    public void printData(){
        System.out.println("Entradas");
        // Imprimir la matriz para verificar
        for (double[] row : inputs) {
            for (double value : row) {
                System.out.print(value + " ");
            }
            System.out.println();
        }

        System.out.println("SALIDAS");
        for (double[] row : outputs) {
            for (double value : row) {
                System.out.print(value + " ");
            }
            System.out.println();
        }




    }


}
