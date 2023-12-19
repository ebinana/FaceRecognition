function w = compute_w(x,U,x_bar,l)

x_centered=x-x_bar;
w=x_centered'*U(:,1:l);

