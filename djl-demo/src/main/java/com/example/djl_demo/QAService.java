package com.example.djl_demo;

import java.nio.file.Paths;

import org.springframework.stereotype.Service;

import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import jakarta.annotation.PostConstruct;

@Service
public class QAService {

    @PostConstruct
    public void loadModel() {

        Criteria<QAInput, String> criteria = Criteria.builder()
                .setTypes(QAInput.class, String.class)
                .optModelPath(
                        Paths.get("/Users/prakashr/Documents/codebase/djl-examples/model/bert-base-cased-squad2/"))
                .optEngine("PyTorch")
                .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                .optProgress(new ProgressBar()).build();

        QAInput qaInput = new QAInput("What is DJL",
                "The Deep Java Library (DJL) is a library developed to help Java developers get started with deep learning. This project is a Spring Boot starter that allows Spring Boot developers to start using DJL for inference.The starter supports dependency management and auto-configuration.");
        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            String result = predictor.predict(qaInput);
            System.out.println("result****  " + result);
        } catch (Exception ex) {
            System.out.println("Exception" + ex);
        }
    }
}
