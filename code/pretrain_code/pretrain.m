function [Ws,bs] = pretrain(X,ds,lambda,beta,rho,max_iter,activation_type)
    nl = size(ds,2);
    % init
    amp = 5e-3;
    for i=1:nl-1
        Ws{i} = amp*randn(ds{i},ds{i+1});
        bs{i} = amp*randn(1,ds{i+1});
    end
    % max_iter>0, using sparse autoencoder to pretrain
    if(max_iter>0)
        % sparse autoencoder pretrain for each layer, except the last output layer
        A = X;
        for i = 1:nl-2
            [Ws{i},bs{i}] = sparse_autoencoder(A,A,ds{i+1},lambda,beta,rho,max_iter,activation_type);
            % hidden layer 
            Z = A*Ws{i}+repmat(bs{i},size(A,1),1);
            A = activation(Z,activation_type);
        end
    end
end
