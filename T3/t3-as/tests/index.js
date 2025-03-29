import assert from "assert";
import { greedy_snake_step } from "../build/release.js";

let out = greedy_snake_step(5, 
    [2,1,2,2,2,3,2,4],
    1,
    [1,1,1,2,1,3,1,4],
    5,
    [5,5,4,5,3,5,2,5,1,5],
    50
)
console.log(out);

out = greedy_snake_step(8, 
    [3, 6, 3, 7, 3, 8, 4, 8],
    3,
    [7, 4, 6, 4, 5, 4, 5, 3, 7, 6, 7, 5, 6, 5, 5, 5, 4, 1, 4, 2, 4, 3, 3, 3],
    10,
    [2, 8, 2, 4, 1, 4, 5, 2, 8, 5, 3, 2, 3, 4, 6, 8, 5, 6, 6, 7],
    100
)

console.log(out);
