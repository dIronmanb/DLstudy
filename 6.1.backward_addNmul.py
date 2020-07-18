from backward_layer import*

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_appleNorange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward propagation 왼쪽 -> 오른쪽
apple_price = mul_apple_layer.forward(apple, apple_num)                 #(1)
orange_price = mul_orange_layer.forward(orange, orange_num)             #(2)
all_price = add_appleNorange_layer.forward(apple_price, orange_price)   #(3)
price = mul_tax_layer.forward(all_price, tax)                           #(4)

print(apple_price, orange_price, all_price, int(price))

#backward propagation 왼쪽 <- 오른쪽
dprice = 1 
dall_price, dtax = mul_tax_layer.backward(dprice)                           #(4)
dapple_price, dorange_price = add_appleNorange_layer.backward(dall_price)   #(3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)             #(2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)                 #(1)

print("dapple_num:%d, dapple:%g, dorange_num:%d, dorange:%g, tax:%d"
      %(int(dapple_num), dapple, int(dorange_num), dorange, dtax))


