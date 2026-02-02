module Test
using ..Model

function predict(chain::Chain, image, label)
    Model.forward(chain, image, label)
    y_true = decode_y_true(chain.cacheList[end].extra[:y_true])
    probs = chain.cacheList[end].extra[:Probs]
    y_pred = decode_y_true(probs)
    return y_true, y_pred
end

end