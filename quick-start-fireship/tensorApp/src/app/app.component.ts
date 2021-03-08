import { Component } from '@angular/core';

import * as tf from '@tensorflow/tfjs';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass']
})
export class AppComponent {

  linearModel: tf.Sequential;
  prediction: any;

  title = 'tensorApp';

  ngOnInit() {
    this.trainNewModel();
  }  

  async trainNewModel() {
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1], useBias: true}));
    this.linearModel.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.adam(0.025, 0.9, 0.999), metrics: [tf.metrics.mse]}); // Maybe tune these hyperparameters

    const xarray: number[] = [...Array(3000).keys()];
    const yarray: number [] = xarray.map((x: number) => 2*x);

    const xs = tf.tensor2d(xarray, [xarray.length, 1]);
    const ys = tf.tensor2d(yarray, [yarray.length, 1]);

    await this.linearModel.fit(xs, ys, { batchSize: 10, epochs: 1000});

    console.log('model trained!')
  }

  linearPrediction(val) {
    val = Number(val);
    const output = this.linearModel.predict(tf.tensor2d([val], [1,1])) as any;
    this.prediction = Array.from(output.dataSync())[0];
  }
}
