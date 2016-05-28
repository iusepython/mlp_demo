function [ a ] = activation( z,type )
    % default: sigmoid
    if nargin == 1
        type = 'sigmoid';
    end 
    % sigmoid function
    if strcmp(type, 'sigmoid')
        a = 1./(1+exp(-z));
    % tanh function
    elseif strcmp(type, 'tanh')
        a = tanh(z);
    % relu function
    elseif strcmp(type, 'relu')
        a = max(0, z)+0.01*min(0,z);
    % softplus function
    elseif strcmp(type, 'softplus')
        a = log(1+exp(z));
    % y=x
    elseif strcmp(type, 'self')
        a = z;
    end
end

