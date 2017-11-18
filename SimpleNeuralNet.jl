#Julia simple neural net for Boston Housing Problem
using Flux.Tracker

cd(@__DIR__)

isfile("housing.data") ||
  download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
           "housing.data")
