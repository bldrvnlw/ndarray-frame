// A minimal version of a dataframe (only column indexes)
// backed by the scijs ndarray for improved speed
const ndarray = require('ndarray');
const concat_col = require('ndarray-concat-cols');
const ops = require('ndarray-ops');
const unsqueeze = require('ndarray-unsqueeze');
const pool = require("ndarray-scratch");
module.exports = class Dataframe {
    /**
     *
     * @param data: a scijs ndarray in row major format where
     *  the number of columns matches the length of the columns array
     * @param columns:
     */
    constructor(data, columns) {
        if (data.dimension != 2) {
            throw new Error('Data must be 2 dimensional');
        }
        if (data.shape[1] != columns.length) {
            throw new Error('Length of columns does not match data');
        }
        this._data = data;
        this._columns = columns;
        this._col_index = {};
        let index = 0;
        for (let col of columns) {
            this._col_index[col] = data.pick(null, index);
            index++;
        }

    }

    get data() {
        return this._data;
    }

    get labels() {
        return this._columns;
    }

    /**
     * Create a new dataframe with shallow references to the original
     * based on the columns selected
     *
     * @param cols - one or more columns to pick. If the final selection is empty
     *      this will return a null. Unfound columns are ignored.
     * @returns {module.dataframescijs || null} - a new dataframescijs or null if no cols match
     */
    sel(...cols) {
        let colVec = [];
        let colIdx = [];
        for (let col in cols) {
            if (this._columns.includes(cols[col])) {
                colVec.push(this._col_index[cols[col]]);
                colIdx.push(cols[col]);
            }
        }
        if (colVec.length === 0) {
            return null;
        }

        let temp = concat_col(colVec);
        return new Dataframe(temp, colIdx);
    }

    /**
     * Return a single column ndarray containing the row mean for the selected cols
     * @param cols
     * @return ndarray
     */
    mean(...cols) {
        let colVec = [];
        for (let col in cols) {
            if (this._columns.includes(cols[col])) {
                colVec.push(this._col_index[cols[col]]);
            }
        }
        let res = ndarray(new Float64Array(this._data.shape[0]));
        if (colVec.length > 0) {
            let temp = concat_col(colVec);
            for (let r = 0; r < temp.shape[0]; r++) {
                res.set(r, ops.sum(temp.pick(r, null))/colVec.length);
            }
        }
        return unsqueeze(pool.clone(res));
    }
    // TODO add more functions
};


