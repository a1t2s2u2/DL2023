class Tensor {
  static shape(arr) {
    const res = [];
    while (Array.isArray(arr)) {
      res.push(arr.length);
      arr = arr[0];
    }
    return res;
  }

  static reshape(arr, shape) {
    arr = arr.flat();
    let res = [];

    let index = 0;
    shape.forEach(dim => {
      res.push(arr.slice(index, index + dim));
      index += dim;
    });
    return res;
  }

  static normalize(arr) {
    let minVal = Number.MAX_VALUE;
    let maxVal = Number.MIN_VALUE;
  
    function findMinMax(array) {
      array.forEach(element => {
        if (Array.isArray(element)) {
          findMinMax(element);
        } else {
          minVal = Math.min(minVal, element);
          maxVal = Math.max(maxVal, element);
        }
      });
    }
  
    function normalize(array) {
      return array.map(element => {
        if (Array.isArray(element)) {
          return normalize(element);
        } else {
          if (minVal === maxVal) {return 0;}
          return (element - minVal) / (maxVal - minVal);
        }
      });
    }
  
    findMinMax(arr);
    return normalize(arr);
  }

  static calc(A, B, operator) {
    if (Array.isArray(B)) {
      const res = [];
      B.forEach((b, i) => {
        const a = Array.isArray(A) ? A[i] : A;
        res.push(Tensor.calc(a, b, operator));
      });
      return res;
    }

    switch (operator) {
      case "add":
        return A + B;
      case "sub":
        return A - B;
      case "mul":
        return A * B;
      case "div":
        return A / B;
      default:
        throw new Error(`Invalid operator: ${operator}`);
    }
  }

  static dot(A, B) {
    let res = 0;
    A.forEach((a, i) => {
      let b = B[i];
      if (Array.isArray(a) || Array.isArray(b)) {
        res += Tensor.dot(a, b);
      } else {
        res += a * b;
      }
    });
    return res;
  }

  static constant(shape, v) {
    if (shape.length === 0) return v;

    const res = [];
    for (let i = 0; i < shape[0]; ++i) {
      res.push(Tensor.constant(shape.slice(1), v));
    }
    return res;
  }

  static zeros(shape) {
    return Tensor.constant(shape, 0);
  }

  static ones(shape) {
    return Tensor.constant(shape, 1);
  }

  static random(shape, min=0, max=1, digit=1) {
    const a = 10**digit;
    if (shape.length === 0) return Math.round( (Math.random() * (max - min) + min)*a ) / a;

    const res = [];
    for (let i = 0; i < shape[0]; ++i) {
      res.push(Tensor.random(shape.slice(1)));
    }
    return res;
  }
}