const expect = require('chai').expect;
const dataframe = require('../src');
const fill = require('ndarray-fill');
const zeros = require('zeros');
const concat_cols = require('ndarray-concat-cols');
const ndtest = require('ndarray-tests');
const ndarray = require('ndarray');
const show = require('ndarray-show');
const unsqueeze = require('ndarray-unsqueeze');
describe('dataframe', function () {

    var rowcol_data = null;
    var label_data = ['A', 'B', 'C', 'D', 'E']; // column labels
    var frame_dims = [10, 5]; // 10 rows , 5 columns
    before(function() {
        rowcol_data = zeros(frame_dims);
        fill(rowcol_data, (i,j) => {
            return 10*i +j;
        });
    });

    it('should instantiate from ndarray and column labels', function () {
        const temp = new dataframe(rowcol_data, label_data);
        expect(temp.data.shape).to.deep.equal(frame_dims);
        expect(temp.labels).to.deep.equal(label_data);
    });

    it('should fail when instantiated with the wrong size ', function() {
        const badInst = function() {
            return new dataframe(rowcol_data, label_data.slice(1,3));
        };
        expect(badInst).to.throw(Error, 'Length of columns does not match data');
    });

    it('should fail when instantiated with data of the wrond dimensions', function() {
        const badInst = function() {
            return new dataframe(zeros([5,5,5]), label_data);
        };
        expect(badInst).to.throw(Error, 'Data must be 2 dimensional');
    });
});

describe('dataframe sel', function () {
    var rowcol_data = null;
    var label_data = ['A', 'B', 'C', 'D', 'E']; // column labels
    var frame_dims = [10, 5]; // 10 rows , 5 columns
    var sub_group = ['B', 'D', 'E'];
    var sub_data = null;
    before(function() {
        rowcol_data = zeros(frame_dims);
        fill(rowcol_data, (i,j) => {
            return 10*i +j;
        })
        sub_data = concat_cols(
            [rowcol_data.pick(null, 1),
            rowcol_data.pick(null, 3),
            rowcol_data.pick(null, 4)]);
    });

    it('should return an ndarray-frame with correct subset', function () {
        const temp = new dataframe(rowcol_data, label_data);
        const result = temp.sel(...sub_group);
        expect(result instanceof dataframe).to.be.true;
        expect(ndtest.equal(result.data, sub_data)).to.be.true;
    });

    it('should return null if no columns match', function () {
        const temp = new dataframe(rowcol_data, label_data);
        const result = temp.sel('Z');
        expect(result).to.be.null;
    });
});

describe('dataframe mean', function () {
    var rowcol_data = null;
    var label_data = ['A', 'B', 'C', 'D', 'E']; // column labels
    var frame_dims = [10, 5]; // 10 rows , 5 columns
    var sub_group = ['A', 'B', 'D', 'E'];
    var sub_data = null;
    var avg_data = zeros([10,1]);
    before(function() {
        rowcol_data = zeros(frame_dims);
        fill(rowcol_data, (i,j) => {
            return 10*i +j;
        });
        sub_data = concat_cols(
            [rowcol_data.pick(null, 0),
            rowcol_data.pick(null, 1),
            rowcol_data.pick(null, 3),
            rowcol_data.pick(null, 4)]);
        for (let r of [...Array(10).keys()]) {
            avg_data.set(r, 0,
                (sub_data.get(r, 0) + sub_data.get(r, 1) + sub_data.get(r, 2) + sub_data.get(r, 3)) / 4);
        }
    });
    it('should return an average of the selected columns', function () {
        const temp = new dataframe(rowcol_data, label_data);
        let result = temp.mean(...sub_group);
        expect(result.shape).to.deep.equal([10,1]);
        // result should be the average of
        //console.log(show(rowcol_data));
        //console.log(show(sub_data));
        //console.log(show(result));
        //console.log(show(avg_data));
        expect(ndtest.equal(avg_data, result)).to.be.true;
    });
    it('should return a zero vector when no columns are selected', function() {
        const temp = new dataframe(rowcol_data, label_data);
        let result = temp.mean('Z');
        expect(ndtest.equal(zeros([10,1]), result)).to.be.true;
    })
});



