# What is Machine Learning?

Something that is not fancy at all, just spatial reasoning.
Machine learning is nothing but a geometry problem.

## Example: Regression

- Type of Supervised Learning
- Fit a line or curve

y = mx + b

## Example: Classification

Type of Supervised Learning
Classification predicts a category/label.
In Binary Classification we use binary categories/labels.
What happened here was to divide the graph into 2 sections using a linear/non-linear line.

## All data is the same
The "meaning" of the numbers is irrelevant to the machine learning model. This is a compelling concept.

## Summary

Take the magic away from machine learning 
Machine learning is just geometry
Regression: Make the line/curve close to the data point
Classification: Make the line/curve separate data points of different classes
All the data is the same.

# Regression Basics

L -> MSE

`m*, b* = arg min L`

We want to minimize the loss with respect to the parameters (m and b)

## Minimize f(x) = x^2

df/dx = 2x = 0
x = 0

## Partial Derivatives

aL/am = 0, aL/ab = 0

Find the derivatives set them to 0, and solve for the parameters

## Problem

This approach of finding the derivative only works for linear regression but not to any other models.
We will use gradient descent instead of derivates.